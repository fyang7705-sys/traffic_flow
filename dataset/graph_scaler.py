from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch


class GraphScaler:

    def __init__(
        self,
        norm_each_channel: bool = False,
        rescale: bool = False,
        stats: Optional[dict] = None,
    ) -> None:
        self.norm_each_channel = norm_each_channel
        self.rescale = rescale
        self.stats = stats if stats is not None else {}

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Fit the scaler to the training data.

        Args:
            data (torch.Tensor): Training data used to fit the scaler.
        """

        # load from previous stats
        if self.stats:
            return
        print("fit data", data.shape)
        # fit from trainining dataset
        if isinstance(data, np.ndarray):
            if self.norm_each_channel:
                mean = np.mean(data, axis=(0, 1), keepdims=True)
                std = np.std(data, axis=(0, 1), keepdims=True)
                std[std == 0] = 1.0  # prevent division by zero by setting std to 1 where it's 0
            else:
                mean = np.mean(data)
                std = np.std(data)
                if std == 0:
                    std = 1.0  # prevent division by zero by setting std to 1 where it's 0
            self.stats['mean'], self.stats['std'] = torch.Tensor(mean), torch.Tensor(std)
            # print("mean shape", self.stats['mean'].shape, "std shape", self.stats['std'].shape)
        else:
            if self.norm_each_channel:
                self.stats['mean'] = torch.mean(data, dim=-2, keepdim=True)
                self.stats['std'] = torch.std(data, dim=-2, keepdim=True)
                self.stats['std'][self.stats['std'] == 0] = 1.0  # prevent division by zero by setting std to 1 where it's 0
            else:
                self.stats['mean'] = torch.mean(data)
                self.stats['std'] = torch.std(data)
                if self.stats['std'] == 0:
                    self.stats['std'] = 1.0  # prevent division by zero by setting std to 1 where it's 0

    def transform(self, input_data: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply Z-score normalization to the input data.

        This method normalizes the input data using the mean and standard deviation computed from the training data. 
        The normalization is applied only to the specified `target_channel`.

        Args:
            input_data (torch.Tensor): The input data to be normalized.
            mask (torch.Tensor, optional): A boolean mask indicating which elements of the input_data should be normalized. 
                If None, all elements are normalized. Defaults to None.

        Returns:
            torch.Tensor: The normalized data with the same shape as the input.
        """
        # print("input shape:", input_data.shape)
        mean = self.stats['mean'].to(input_data.device)
        std = self.stats['std'].to(input_data.device)
        normed_data = (input_data - mean) / std
        if mask is not None:
            normed_data = torch.where(mask, normed_data, input_data)
        return normed_data

    def inverse_transform(self, input_data: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reverse the Z-score normalization to recover the original data scale.

        This method transforms the normalized data back to its original scale using the mean and standard deviation 
        computed from the training data. This is useful for interpreting model outputs or for further analysis in the original data scale.

        Args:
            input_data (torch.Tensor): The normalized data to be transformed back.
            mask (torch.Tensor, optional): A boolean mask indicating which elements of the input_data should be transformed back. 
                If None, all elements are transformed back. Defaults to None.

        Returns:
            torch.Tensor: The data transformed back to its original scale.
        """

        mean = self.stats['mean'].to(input_data.device)
        std = self.stats['std'].to(input_data.device)
        denormed_data = input_data * std + mean
        if mask is not None:
            denormed_data = torch.where(mask, denormed_data, input_data)
        return denormed_data
