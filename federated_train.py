import os
from pathlib import Path

import torch
import wandb
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader


from datasets.build import build_dataset, build_datasets
from models.build import build_encoder
from servers.build import get_server_type, build_server
from clients.build import get_client_type
from evalers.build import get_evaler_type
# from trainer.build import build_trainer
# from trainers import Trainer

from trainers.build import get_trainer_type

from utils import initalize_random_seed

import hydra
from omegaconf import DictConfig
import omegaconf
import coloredlogs, logging

# import loggings
logger = logging.getLogger(__name__)
custom_level_styles = {'info': {'color': 'yellow', 'bold': True}}  # info 레벨의 색상을 노란색으로 변경
# coloredlogs.install(fmt='%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s')
coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S',
                    level_styles=custom_level_styles)

wandb.require("service")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args : DictConfig) -> None:

    #args = parse_arguments().parse_args()
    if args.multiprocessing:
        torch.multiprocessing.set_sharing_strategy('file_system')
        set_start_method('spawn', True)
    pid = os.getpid()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_device)

    args.log_dir = Path(args.log_dir)
    #args.CHECKPOINT_DIR = Path(args.log_dir)
    # exp_name = args.method_name if args.remark == "" else f"{args.method_name}_{args.remark}"
    exp_name = args.exp_name if args.remark == "" else f"{args.exp_name}_{args.remark}"
    args.log_dir = args.log_dir / args.dataset.name / f'{args.split.mode}{str(args.split.alpha) if args.split.mode == "dirichlet" else ""}' /exp_name
    print(exp_name)
    if not args.log_dir.exists():
        args.log_dir.mkdir(parents=True, exist_ok=True)

    ## Wandb

    if args.wandb:
        if args.get('wandb_resume_id'):
            wandb.init(entity='snow1234',
                    project=args.project,
                    group=f'{args.split.mode}{str(args.split.alpha) if args.split.mode == "dirichlet" else ""}',
                    job_type=exp_name,
                    dir=args.log_dir,
                    notes="PID: " + str(pid),
                    id=args.wandb_resume_id,
                    resume='must',
                    )
            wandb.run.name = exp_name
            wandb.config.update(omegaconf.OmegaConf.to_container(
                args, resolve=True, throw_on_missing=True
            ), allow_val_change=True)
            # wandb.config.update(omegaconf.OmegaConf.to_container(
            #     args, resolve=True, throw_on_missing=True
            # ))
        else:
            wandb.init(entity='snow1234',
                    project=args.project,
                    group=f'{args.split.mode}{str(args.split.alpha) if args.split.mode == "dirichlet" else ""}',
                    job_type=exp_name,
                    dir=args.log_dir,
                    notes="PID: " + str(pid))
            wandb.run.name = exp_name
            # wandb.run.save()
            # wandb.config.update(args)
            wandb.config.update(omegaconf.OmegaConf.to_container(
                args, resolve=True, throw_on_missing=True
            ))

    initalize_random_seed(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # datasets = build_datasets(args)
    model = build_encoder(args)
    client_type = get_client_type(args)
    server = build_server(args)
    datasets = build_datasets(args)
    # get the num_classes from dataset, and set it to the fc layer of the model.
    evaler_type = get_evaler_type(args)

    # trainer = Trainer(model=model, client_type=client_type, server=server, evaler_type=evaler_type,
    #                   datasets=datasets,
    #                   device=device, args=args, config=None)
    trainer_type = get_trainer_type(args)
    trainer = trainer_type(model=model, client_type=client_type, server=server, evaler_type=evaler_type,
                           datasets=datasets,
                           device=device, args=args, config=None)
    trainer.train()

    

if __name__ == '__main__':
    main()