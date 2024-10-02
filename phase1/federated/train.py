import os
import argparse

import numpy as np
import torch
import shutil
import random

from torch import nn, optim
from tqdm import tqdm

from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from factory.clip import CLIP

import warnings
warnings.filterwarnings("ignore")

def partial_load(ckpt, state_dict):
    for k, v in list(ckpt.items()):
        if k not in state_dict:
            ckpt.pop(k)
        elif type(v) == dict:
            ckpt[k] = partial_load(ckpt[k], state_dict[k])
        elif type(v) == torch.Tensor:
            if v.shape != state_dict[k].shape:
                ckpt.pop(k)
    state_dict.update(ckpt)
    return state_dict

class MyStrategy(DefaultStrategy):
    def sample_nodes(self, round_i):
        sampled_nodes = self._fds.node_ids()
        if len(sampled_nodes) > 2:
            sampled_nodes = random.sample(sampled_nodes, 2)
        self._sampling_node_history[round_i] = sampled_nodes
        return sampled_nodes

class MyRemoteTrainingPlan(TorchTrainingPlan):
    def __init__(self):
        super().__init__()
        self.idx = -1
        self.done = False

    def init_training(self):
        self.device = next(self.net.parameters()).device
        
        node_id = self.args.node_id
        if os.path.isdir(f"sample_{node_id}"):
            last_sample = int(sorted(os.listdir(f"sample_{node_id}"))[-1].replace("_real", "").replace(".png", ""))
            round_current = last_sample // self.args.num_updates_per_round + 1
            start_iter = round_current * self.args.num_updates_per_round
        else:
            start_iter = 0

        self.pbar = iter(range(start_iter, self.args.iter))
        self.pbar = tqdm(self.pbar, initial=start_iter, total=self.args.iter, dynamic_ncols=True, smoothing=0.01)
        
        self.loss_dict = {}
        if self.args.augment and self.args.augment_p == 0:
            self.ada_augment = AdaptiveAugment(self.args.ada_target, self.args.ada_length, 8, self.device)
        
        self.mean_path_length = 0
        self.d_loss_val = 0
        self.r1_loss = torch.tensor(0.0, device=self.device)
        self.g_loss_val = 0
        self.path_loss = torch.tensor(0.0, device=self.device)
        self.path_lengths = torch.tensor(0.0, device=self.device)
        self.mean_path_length_avg = 0

        self.ada_aug_p = self.args.augment_p if self.args.augment_p > 0 else 0.0
        self.r_t_stat = 0
        self.accum = 0.5 ** (32 / (10 * 1000))

        self.sample_labels = torch.tensor(range(self.args.n_domains)).repeat(self.args.n_sample)# TODO: generalize n. of labels
        self.sample_labels = torch.nn.functional.one_hot(self.sample_labels, num_classes=self.args.n_domains).float().to(self.device)# TODO: generalize n. of labels
        self.sample_zaxes = get_random_labels(len(self.sample_labels), 29, self.device)
        self.sample_labels = torch.cat([self.sample_labels, self.sample_zaxes], dim=1)
        torch.manual_seed(0); self.sample_z = torch.randn(
            self.args.n_sample, self.args.latent, device=self.device
        ).repeat_interleave(self.args.n_domains, dim=0)
        self.real_sample = None
        self.real_sample_labels = None
        self.domain_label = None

        if self.net.clip_loss.clip_mean is None:
            print("Creating new instance of clip loss")
            self.net.clip_loss.load_mean(
                os.path.join(os.path.dirname(self.dataset_path), "clip_mean.npy"), 
                self.args.n_domains * self.args.n_sample,
            )
        self.net.clip_loss = self.net.clip_loss.to(self.device)


    def init_model(self, model_args):
        self.args = DictToObject(model_args)

        generator = Generator(
            self.args.size, self.args.latent, self.args.n_mlp, self.args.n_domains, channel_multiplier=self.args.channel_multiplier,
        )
        discriminator = Discriminator(
            self.args.size, self.args.n_domains, channel_multiplier=self.args.channel_multiplier
        )
        g_ema = Generator(
            self.args.size, self.args.latent, self.args.n_mlp, self.args.n_domains, channel_multiplier=self.args.channel_multiplier
        )
        g_ema.eval()
        accumulate(g_ema, generator, 0)

        self.net = self.Net(generator, discriminator, g_ema, model_args)
        return self.net
    
    def init_optimizer(self):
        g_reg_ratio = self.args.g_reg_every / (self.args.g_reg_every + 1)
        d_reg_ratio = self.args.d_reg_every / (self.args.d_reg_every + 1)

        g_optim = optim.Adam(
            self.net.generator.parameters(),
            lr=self.args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        d_optim = optim.Adam(
            self.net.discriminator.parameters(),
            lr=self.args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        self.myOptimizer = self.MyOptimizer(g_optim, d_optim)
        return self.myOptimizer
    
    def init_dependencies(self):
        return [
            "import os",
            "from torch import optim",
            "from tqdm import tqdm",
            "from torchvision import utils",
            "from factory.utils import *",
            "from factory.distributed import get_rank, reduce_loss_dict, reduce_sum, get_world_size",
            "from factory.non_leaking import AdaptiveAugment, augment",
            "from factory.model import Generator, Discriminator",
            "from factory.dataset import MultiResolutionDataset",
            "from factory.clip import CLIP",
        ]

    class Net(nn.Module):
        class CLIPLoss():
            def __init__(self, clip_weight):
                self.clip = CLIP() if clip_weight > 0 else None
                self.clip_mean = None
                self.clip_weight = clip_weight
            
            def to(self, device):
                self.clip_mean = self.clip_mean.to(device)
                self.clip = self.clip.to(device) if self.clip_weight > 0 else None
                return self
            
            def load_mean(self, mean_path, repeat):
                self.clip_mean = torch.tensor(np.load(mean_path)).float()
                self.clip_mean = self.clip_mean.repeat(repeat, 1)
            
            def __call__(self, fake_img, labels):
                labels, encodings = labels
                assert torch.all(labels[0, :3] == labels[:, :3])

                def spherical_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    x = nn.functional.normalize(x, dim=-1)
                    y = nn.functional.normalize(y, dim=-1)
                    return (x * y).sum(-1).arccos().pow(2)
                
                gen_img_features = self.clip.encode_image(fake_img)
                return self.clip_weight * spherical_distance(gen_img_features, encodings).mean()


        def __init__(self, generator, discriminator, g_ema, model_args: dict = {}):
            super().__init__()
            self.generator = generator
            self.discriminator = discriminator
            self.g_ema = g_ema
            self.clip_loss = self.CLIPLoss(model_args["clip_weight"])

        def forward(self):
            return None

        def load_state_dict(self, state_dict,
                        strict: bool = True):
            def partial_load(ckpt, state_dict):
                for k, v in list(ckpt.items()):
                    if k not in state_dict:
                        ckpt.pop(k)
                    elif type(v) == dict:
                        ckpt[k] = partial_load(ckpt[k], state_dict[k])
                    elif type(v) == torch.Tensor:
                        if v.shape != state_dict[k].shape:
                            ckpt.pop(k)
                state_dict.update(ckpt)
                return state_dict
            state_dict = partial_load(state_dict, self.state_dict())
            return super().load_state_dict(state_dict, strict=strict)
    
    class MyOptimizer(optim.Optimizer):
        def __init__(self, g_optim, d_optim):
            self.g_optim = g_optim
            self.d_optim = d_optim
    
        def zero_grad(self, set_to_none: bool = False):
            self.g_optim.zero_grad(set_to_none=set_to_none)
            self.d_optim.zero_grad(set_to_none=set_to_none)
        
        def switch_optimizer(self, idx):
            self.idx = idx
            if idx % 2:
                self.param_groups = self.g_optim.param_groups
            else:
                self.param_groups = self.d_optim.param_groups
        
        def __setstate__(self, state):   
            if self.idx % 2:
                self.g_optim.__setstate__(state)
            else:
                self.d_optim.__setstate__(state)

        @torch.no_grad()
        def step(self, closure=None):
            if self.idx % 2:
                self.g_optim.step(closure)
            else:
                self.d_optim.step(closure)

    def log_results(self, loss_reduced):
        i = self.idx

        self.d_loss_val = loss_reduced["d"].mean().item() if "d" in loss_reduced else 0
        self.g_loss_val = loss_reduced["g"].mean().item() if "g" in loss_reduced else 0
        r1_val = loss_reduced["r1"].mean().item() if "r1" in loss_reduced else 0
        path_loss_val = loss_reduced["path"].mean().item() if "path" in loss_reduced else 0
        real_score_val = loss_reduced["real_score"].mean().item() if "real_score" in loss_reduced else 0
        fake_score_val = loss_reduced["fake_score"].mean().item() if "fake_score" in loss_reduced else 0
        path_length_val = loss_reduced["path_length"].mean().item() if "path_length" in loss_reduced else 0

        self.pbar.set_description(
            (
                f"d: {self.d_loss_val:.4f}; g: {self.g_loss_val:.4f}; r1: {r1_val:.4f}; "
                f"path: {path_loss_val:.4f}; mean path: {self.mean_path_length_avg:.4f}; "
                f"augment: {self.ada_aug_p:.4f}; domain: {self.domain_label}"
            )
        )

        node_id = self.args.node_id
        if not os.path.isdir(f"sample_{node_id}"):
            os.makedirs(f"sample_{node_id}")

        if i % 100 == 0:
            with torch.no_grad():
                self.net.g_ema.eval()
                sample, _ = self.net.g_ema(
                    [self.sample_z], (
                        self.sample_labels,
                        None if random.random() < .5 else self.net.clip_loss.clip_mean
                    ), randomize_noise=False
                )
                utils.save_image(
                    sample,
                    f"sample_{node_id}/{str(i).zfill(6)}.png",
                    nrow=self.args.n_sample,
                    normalize=True,
                    range=(-1, 1),
                )
                if self.real_sample is not None:
                    utils.save_image(
                        self.real_sample,
                        f"sample_{node_id}/{str(i).zfill(6)}_real.png",
                        nrow=len(self.real_sample),
                        normalize=True,
                        range=(-1, 1),
                    )

    def discriminator_step(self, real_img, real_labels):
        noise = mixing_noise(min(self.args.batch, len(real_labels[0])), self.args.latent, self.args.mixing, self.device)
        fake_img, _ = self.net.generator(noise, real_labels)

        if self.args.augment:
            real_img_aug, _ = augment(real_img, self.ada_aug_p)
            fake_img, _ = augment(fake_img, self.ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = self.net.discriminator(fake_img, real_labels)
        real_pred = self.net.discriminator(real_img_aug, real_labels)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        self.loss_dict["d"] = d_loss
        self.loss_dict["real_score"] = real_pred.mean()
        self.loss_dict["fake_score"] = fake_pred.mean()

        #self.net.discriminator.zero_grad()
        #d_loss.backward()
        #self.optimizer.d_optim.step()

        if self.args.augment and self.args.augment_p == 0:
            self.ada_aug_p = self.ada_augment.tune(real_pred)
            self.r_t_stat = self.ada_augment.r_t_stat

        return d_loss

    def d_regularize_step(self, real_img, real_labels):
        real_img.requires_grad = True

        if self.args.augment:
            real_img_aug, _ = augment(real_img, self.ada_aug_p)

        else:
            real_img_aug = real_img

        real_pred = self.net.discriminator(real_img_aug, real_labels)
        self.r1_loss = d_r1_loss(real_pred, real_img)

        self.loss_dict["r1"] = self.r1_loss

        #self.net.discriminator.zero_grad()
        #(args.r1 / 2 * self.r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
        #self.optimizer.d_optim.step()

        return self.args.r1 / 2 * self.r1_loss * self.args.d_reg_every + 0 * real_pred[0]

    def generator_step(self, real_labels):
        noise = mixing_noise(min(self.args.batch, len(real_labels[0])), self.args.latent, self.args.mixing, self.device)
        fake_img, _ = self.net.generator(noise, real_labels)

        if self.args.augment:
            fake_img, _ = augment(fake_img, self.ada_aug_p)

        fake_pred = self.net.discriminator(fake_img, real_labels)
        g_loss = g_nonsaturating_loss(fake_pred)
        if self.net.clip_loss.clip_weight > 0 and real_labels[1] is not None:
            g_loss += self.net.clip_loss(fake_img, real_labels)

        self.loss_dict["g"] = g_loss

        #self.net.generator.zero_grad()
        #g_loss.backward()
        #self.optimizer.g_optim.step()

        return g_loss

    def g_regularize_step(self, real_labels):
        path_batch_size = max(1, self.args.batch // self.args.path_batch_shrink)
        noise = mixing_noise(min(path_batch_size, len(real_labels[0])), self.args.latent, self.args.mixing, self.device)
        fake_img, latents = self.net.generator(noise, (
            real_labels[0][:path_batch_size], real_labels[1][:path_batch_size] if real_labels[1] is not None else None
        ), return_latents=True)

        self.path_loss, self.mean_path_length, self.path_lengths = g_path_regularize(
            fake_img, latents, self.mean_path_length
        )

        self.loss_dict["path"] = self.path_loss
        self.loss_dict["path_length"] = self.path_lengths.mean()

        weighted_path_loss = self.args.path_regularize * self.args.g_reg_every * self.path_loss
        if self.args.path_batch_shrink:
            weighted_path_loss += 0 * fake_img[0, 0, 0, 0]
        
        #self.net.generator.zero_grad()
        #weighted_path_loss.backward()
        #self.optimizer.g_optim.step()

        self.mean_path_length_avg = (
            reduce_sum(self.mean_path_length).item() / get_world_size()
        )

        return weighted_path_loss


    def training_step(self, real_img, target):
        if self.idx == -1:
            self.init_training()
        if type(self.pbar) == tqdm:
            if self.idx > -1:
                self.pbar.update(1)
            self.idx = self.pbar.n
        else:
            self.idx = next(self.pbar)
        
        self.myOptimizer.switch_optimizer(self.idx)
        i = self.idx#idx + args.start_iter

        torch.cuda.empty_cache()
        loss_reduced = reduce_loss_dict(self.loss_dict) if i > 0 else {}
        self.log_results(loss_reduced)

        if i > self.args.iter:
            self.done = True
            return

        torch.cuda.empty_cache()
        real_labels, real_zaxes = target
        if self.domain_label is not None:
            assert torch.all(real_labels == self.domain_label).item()
        self.domain_label = real_labels[0].item()
        
        real_img = real_img.to(self.device)
        real_labels = torch.nn.functional.one_hot(real_labels, num_classes=self.args.n_domains).float().to(self.device)# TODO: generalize n. of labels
        real_zaxes = torch.nn.functional.one_hot(real_zaxes, num_classes=29).float().to(self.device)
        real_labels = torch.cat([real_labels, real_zaxes], dim=1)
        if random.random() < 0.5:
            real_encodings = None
        else:
            real_encodings = self.net.clip_loss.clip_mean[:len(real_labels)]#self.net.clip_loss.clip.encode_image(real_img)
        real_labels = (real_labels, real_encodings)
        
        step_loss = 0

        #According to idx I have to decide which loss to calculate.
        #Before, at each iteration we used to backpropagate at least two losses + 2 additional
        #Now idx is doubled and sometimes 3x or 4x than the old one.

        if i % 2 == 0:
            requires_grad(self.net.generator, False)
            requires_grad(self.net.discriminator, True)
            d_regularize = (i // 2) % self.args.d_reg_every == 0
            if not d_regularize:
                step_loss = self.discriminator_step(real_img, real_labels)
            else:
                step_loss = self.d_regularize_step(real_img, real_labels)

        #torch.cuda.empty_cache()

        else:
            requires_grad(self.net.generator, True)
            requires_grad(self.net.discriminator, False)
            g_regularize = (i // 2) % self.args.g_reg_every == 0
            if not g_regularize:
                step_loss = self.generator_step(real_labels)
            else:
                step_loss = self.g_regularize_step(real_labels)
            accumulate(self.net.g_ema, self.net.generator, self.accum)            
        
        self.real_sample = real_img
        self.real_sample_labels = real_labels
        
        return step_loss

    def training_data(self):
        transform = transforms.Compose(
            [
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        
        dataset = MultiResolutionDataset(self.dataset_path, transform, self.args.size)

        #TODO: add Sampler
        return DataManager(dataset,  shuffle=True, drop_last=True)


def train():
    tags =  ["#im2im"]

    model_args = {**args.__dict__}

    training_args = {
        'loader_args': {
            'batch_size': args.batch,
        },
        #'epochs': args.iter,
        'num_updates': args.num_updates_per_round, 
        'dry_run': False,
        'use_gpu': True,
    }

    # federated learning training
    if any([p.startswith("sample_") and os.path.isdir(p) for p in os.listdir("./")]):
        exp = Experiment.load_breakpoint()
        exp.set_model_args(model_args)
        exp.set_training_plan_class(MyRemoteTrainingPlan)
        exp.set_training_args(training_args)
        exp.set_round_limit(args.rounds)
        exp.set_strategy(MyStrategy)
        exp._job._model_args = exp._model_args
        exp._job._training_args = exp._training_args
        exp._job._training_plan_class = exp._training_plan_class

    else:
        exp = Experiment(
            tags=tags,
            model_args=model_args,
            training_plan_class=MyRemoteTrainingPlan,
            training_args=training_args,
            round_limit=args.rounds,
            aggregator=FedAverage(),
            node_selection_strategy=MyStrategy,
            save_breakpoints=True,
        )
    exp.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--rounds", type=int, default=350, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--partial_load", action="store_true", help="use partial checkpoints"
    )
    parser.add_argument(
        "--transfer_learning", action="store_true", help="perform transfer learning, i.e., re-initialize optimazers"
    )
    parser.add_argument(
        "--num_updates_per_round", type=int, default=2000, help="total training iterations"
    )
    parser.add_argument(
        "--n_domains", type=int, default=2, help="total number of domains"
    )
    parser.add_argument(
        "--clip-weight", type=float, default=0.2, help="total number of domains"
    )

    args = parser.parse_args()

    args.n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    
    args.latent = 512
    args.n_mlp = 8

    args.iter = 350 * 2000 + 35 * 200#args.rounds * args.num_updates_per_round

    train()
