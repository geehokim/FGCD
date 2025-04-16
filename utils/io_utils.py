

from typing import Union, Any, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path


import logging
logger = logging.getLogger(__name__)


def load_checkpoint(model_path: Path, jit: bool = True) -> Dict:
    checkpoint = {}
    model = torch.jit.load(str(model_path), map_location='cpu')
    checkpoint['model'] = model
    # if jit:
    #     model = torch.jit.load(str(model_path))
    #     checkpoint['model'] = model
    # else:
    #     checkpoint = torch.load(model_path)
    return checkpoint



def save_checkpoint(model: Union[nn.Module, torch.jit.ScriptModule],
                    # optimizer: Union[nn.Module, torch.jit.ScriptModule],
                    local_models,
                    prev_global_model_state,
                    output_model_path: Path,
                    epoch: int,
                    save_torch: bool = False,
                    use_breakpoint: bool = False,
                    ) -> None:

    # device = next(model.parameters()).device
    # print("save checkpoint device : ", get_rank(), device)

    if not output_model_path.parent.exists():
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
    # model = prepare_model_for_export(model, config=config)

    model_script = None
    try:
        model_script = torch.jit.script(model)
    except Exception as e:
        print(f"======================================= {e}")
        logger.error("Error during jit scripting, so save torch state_dict.")
        if use_breakpoint:
            breakpoint()
        # if get_rank() == 0:
        # save_torch_model(model=model, optimizer=optimizer, output_model_path=output_model_path, epoch=epoch)

    save_path = None
    # if get_rank() == 0:
    if model_script is not None:
        try:
            save_path = output_model_path.parent / f'{output_model_path.name}.jit'
            torch.jit.save(model_script, str(save_path))
            logger.warning(f"Saved torchscript model at {save_path}")
        except:
            breakpoint()
            # save_path = path_numbering(output_model_path)
            # torch.jit.save(model_script, str(save_path))
            # logger.warning("Saved torchscript model at {}".format(save_path))

    if save_torch:
        save_torch_path = output_model_path
        # save_torch_path = f'{output_model_path.name}.torch.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'local_models_dict': local_models,
            'prev_state_dict': prev_global_model_state,
        }, save_torch_path)
        logger.warning(f"Saved torch model at {save_torch_path}")

    return save_path






def save_checkpoint_momentum(model: Union[nn.Module, torch.jit.ScriptModule],
                # optimizer: Union[nn.Module, torch.jit.ScriptModule],
                output_model_path: Path,
                epoch: int,
                delta,
                momentum,
                save_torch: bool = False,
                use_breakpoint: bool = False,
                ) -> None:

    # device = next(model.parameters()).device
    # print("save checkpoint device : ", get_rank(), device)

    if not output_model_path.parent.exists():
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
    # model = prepare_model_for_export(model, config=config)

    model_script = None
    try:
        model_script = torch.jit.script(model)
    except Exception as e:
        print(f"======================================= {e}")
        logger.error("Error during jit scripting, so save torch state_dict.")
        if use_breakpoint:
            breakpoint()
        # if get_rank() == 0:
        # save_torch_model(model=model, optimizer=optimizer, output_model_path=output_model_path, epoch=epoch)

    save_path = None
    # if get_rank() == 0:
    if model_script is not None:
        try:
            save_path = output_model_path.parent / f'{output_model_path.name}.jit'
            torch.jit.save(model_script, str(save_path))
            logger.warning(f"Saved torchscript model at {save_path}")
        except:
            breakpoint()
            # save_path = path_numbering(output_model_path)
            # torch.jit.save(model_script, str(save_path))
            # logger.warning("Saved torchscript model at {}".format(save_path))

    if save_torch:
        save_torch_path = output_model_path
        # save_torch_path = f'{output_model_path.name}.torch.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'delta' : delta,
            'momentum' : momentum,
        }, save_torch_path)
        logger.warning(f"Saved torch model at {save_torch_path}")

    return save_path