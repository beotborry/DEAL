"""
Code based on
https://github.com/rwightman/pytorch-image-models.git
"""

import torch
import torch.optim as optim


def _get_optim_n_scheduler(args, model, num_iters=None):
    if isinstance(model, tuple):
        for i in range(len(model)):
            if i == 0:
                param_list = list(model[i].parameters())
            else:
                param_list = param_list + list(model[i].parameters())
    else:
        param_list = model.parameters()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(param_list, momentum=0.9,
                              lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    return optimizer, scheduler