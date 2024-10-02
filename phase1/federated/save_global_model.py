from fedbiomed.fedbiomed.researcher.experiment import Experiment
import train
import torch
from train import MyStrategy

if __name__ == "__main__":
    exp = Experiment.load_breakpoint()
    local_training_plan = train.MyRemoteTrainingPlan()
    
    for dependency_statement in local_training_plan.init_dependencies():
        print(dependency_statement)
        exec(dependency_statement)
    train.DictToObject = DictToObject
    train.Generator = Generator
    train.Discriminator = Discriminator
    train.accumulate = accumulate
    
    local_model = local_training_plan.init_model(exp.model_args())
    local_model.load_state_dict(exp.aggregated_params()[exp.round_current()-1]['params'])

    torch.save({
            "g": local_model.generator.state_dict(),
            "d": local_model.discriminator.state_dict(),
            "g_ema": local_model.g_ema.state_dict(),
        },
        f"final_global_model.pt",
    )
