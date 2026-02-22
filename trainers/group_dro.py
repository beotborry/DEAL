import random
from typing import Iterator, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Sampler

from utils import set_seed
from typing import Callable


class CustomIndicesSampler(Sampler):
    """
    Samples from the specified indices (pass indices - upsampled, downsampled, group balanced etc. to this class)
    Default is no shuffle.
    """
    def __init__(
        self, 
        indices: List[int],
        shuffle: bool = False,
        seed: int = 0,
    ):
        """
        Samples elements from the specified indices.

        :param indices: The list of indices to sample from.
        :type indices: list[int]
        :param shuffle: Whether to shuffle the indices. Default is False.
        :type shuffle: bool, optional
        """
         
        set_seed(seed)
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over the sampled indices.

        :return: An iterator over the sampled indices.
        :rtype: iterator[int]
        """
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self) -> int:
        """
        Returns the number of sampled indices.

        :return: The number of sampled indices.
        :rtype: int
        """
        return len(self.indices)

class GroupWeightedLoss(nn.Module):
    """
    A module for computing group-weighted loss.
    """
    def __init__(
        self, 
        criterion: Callable[[torch.tensor, torch.tensor], torch.tensor],
        num_groups: int,
        group_weight_lr: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initializes GroupWeightedLoss.

        :param criterion: The loss criterion function.
        :type criterion: Callable[[torch.tensor, torch.tensor], torch.tensor]
        :param num_groups: The number of groups to consider.
        :type num_groups: int
        :param group_weight_lr: The learning rate for updating group weights (default: 0.01).
        :type group_weight_lr: float
        :param device: The device on which to perform computations. Defaults to CPU.
        :type device: torch.device
        """
        super(GroupWeightedLoss, self).__init__()
        self.criterion = criterion
        self.device = device
        self.num_groups = num_groups
        self.group_weights = torch.ones(self.num_groups).to(self.device)
        self.group_weights.data = self.group_weights.data / self.group_weights.data.sum()
        self.group_weight_lr = group_weight_lr

    def forward(self, outputs, labels, groups):
        """
        Computes the group-weighted loss.
        """
        # compute loss for different groups and update group weights
        loss = self.criterion(outputs, labels)
        group_loss = torch.zeros(self.num_groups).to(self.device)
        for i in range(self.num_groups):
            if (groups==i).sum() > 0:
                group_loss[i] += loss[groups==i].mean()
        self.update_group_weights(group_loss)

        # compute weighted loss
        loss = group_loss * self.group_weights
        loss = loss.sum()
        
        return loss

    def update_group_weights(self, group_loss):
        group_weights = self.group_weights
        group_weights = group_weights * torch.exp(self.group_weight_lr * group_loss)
        group_weights = group_weights / group_weights.sum()
        self.group_weights.data = group_weights.data