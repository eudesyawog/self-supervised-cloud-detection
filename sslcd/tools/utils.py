import random
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data._utils.collate import default_collate

def seed_all(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

def remove_argument(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            if group_action.dest == arg:
                action._group_actions.remove(group_action)
                return

def modify_choices(parser, dest, choices):
    for action in parser._actions:
        if action.dest == dest:
            action.choices = choices
            action.default = choices[0]
            return
    else:
        raise AssertionError("argument {} not found".format(dest))

def remove_bias_and_norm_from_weight_decay(parameter_groups):
    out = []
    for group in parameter_groups:
        # parameters with weight decay
        decay_group = {k: v for k, v in group.items() if k != "params"}

        # parameters without weight decay
        no_decay_group = {k: v for k, v in group.items() if k != "params"}
        no_decay_group["weight_decay"] = 0
        group_name = group.get("name", None)
        if group_name:
            no_decay_group["name"] = group_name + "_no_decay"

        # split parameters into the two lists
        decay_params = []
        no_decay_params = []
        for param in group["params"]:
            if param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # add groups back
        if decay_params:
            decay_group["params"] = decay_params
            out.append(decay_group)
        if no_decay_params:
            no_decay_group["params"] = no_decay_params
            out.append(no_decay_group)
    return out


class IndexedDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        
    def __getitem__(self, index):
        data = self.dataset[index]
        return (torch.tensor(index), *data)

    def __len__(self):
        return len(self.dataset)