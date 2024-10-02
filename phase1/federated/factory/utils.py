import torch
import random
import math
import numpy as np
from torch import autograd
from torch.nn import functional as F
from a2v_fed.op import conv2d_gradfix

class DictToObject:
    def __init__(self, dictionary):
        self.__dict__ = dictionary

    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

def get_random_labels(batch, n_labels, device):
    labels = (torch.rand(batch)*n_labels).long()
    labels = torch.nn.functional.one_hot(labels, num_classes=n_labels).float().to(device)
    return labels

def get_fixed_labels(label, batch, n_labels, device):
    labels = torch.tensor([label]*batch, dtype=torch.long)
    labels = torch.nn.functional.one_hot(labels, num_classes=n_labels).float().to(device)
    return labels

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises

def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths