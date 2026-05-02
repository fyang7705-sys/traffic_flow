import os
from typing import Union

import numpy as np

from basicts.utils.constants import BasicTSMode

from basicts.data.base_dataset import BasicTSDataset


class HybridPIBTDataset(BasicTSDataset):
    """Hybrid PIBT forecasting dataset.

    支持两种数据形状：
    1) 单条长序列: data.shape == (T, F)
    2) 多条短序列: data.shape == (S, T, F)（每条序列长度相同）

    对于 (S, T, F) 的情况：
    - 每条序列可采样窗口数 n_per_series = T - (input_len + output_len) + 1
    - __getitem__(index) 的 index 是全局窗口索引，会映射为：
        series_id = index // n_per_series
        start_t   = index %  n_per_series
    """

    def __init__(
            self,
            dataset_name: str,
            input_len: int,
            output_len: int,
            mode: Union[BasicTSMode, str],
            use_timestamps: bool = False,
            local: bool = True,
            data_file_path: Union[str, None] = None,
            memmap: bool = False) -> None:
        super().__init__(dataset_name, mode, memmap)
        self.input_len = input_len
        self.output_len = output_len

        if not local:
            pass  # TODO: support download remotely
        if data_file_path is None:
            data_file_path = f"datasets/{dataset_name}"  # default file path

        try:
            self._data = np.load(
                os.path.join(data_file_path, f"{mode}_data.npy"),
                mmap_mode="r" if memmap else None,
            )
            if use_timestamps:
                self.timestamps = np.load(
                    os.path.join(data_file_path, f"{mode}_timestamps.npy"),
                    mmap_mode="r" if memmap else None,
                )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Cannot load dataset from {data_file_path}, Please set a correct local path."
                "If you want to download the dataset, please set the argument `local` to False."
            ) from e

        self.memmap = memmap
        self.use_timestamps = use_timestamps

        # Determine mode: single-series (T,F) vs multi-series (S,T,F)
        if not isinstance(self._data, np.ndarray) or self._data.ndim not in (2, 3):
            raise ValueError(f"Expected data shape (T,F) or (S,T,F), got {getattr(self._data, 'shape', None)}")

        self._multi_series = self._data.ndim == 3
        self._win = self.input_len + self.output_len

        if self._multi_series:
            # data: (S,T,F)
            self._num_series = int(self._data.shape[0])
            self._series_len = int(self._data.shape[1])
            self._n_per_series = self._series_len - self._win + 1
            if self._n_per_series <= 0:
                raise ValueError(
                    f"Series length T={self._series_len} is too short for input_len+output_len={self._win}."
                )

            if self.use_timestamps:
                if not isinstance(self.timestamps, np.ndarray) or self.timestamps.ndim < 2:
                    raise ValueError(f"Expected timestamps to be ndarray, got {type(self.timestamps)}")
                if self.timestamps.shape[0] != self._num_series or self.timestamps.shape[1] != self._series_len:
                    raise ValueError(
                        "timestamps shape must match data on first two dims (S,T). "
                        f"data={self._data.shape}, timestamps={self.timestamps.shape}"
                    )

    def __getitem__(self, index: int) -> dict:
        item = {}

        if not self._multi_series:
            # data: (T,F)
            history_data = self._data[index: index + self.input_len]
            future_data = self._data[index + self.input_len: index + self.input_len + self.output_len]
            # keep a consistent shape [T, N, C]
            history_data = history_data[..., np.newaxis]
            future_data = future_data[..., np.newaxis]
            item["inputs"] = history_data.copy() if self.memmap else history_data
            item["targets"] = future_data.copy() if self.memmap else future_data
            if self.use_timestamps:
                history_timestamps = self.timestamps[index: index + self.input_len]
                future_timestamps = self.timestamps[index + self.input_len: index + self.input_len + self.output_len]
                item["inputs_timestamps"] = history_timestamps.copy() if self.memmap else history_timestamps
                item["targets_timestamps"] = future_timestamps.copy() if self.memmap else future_timestamps
            return item

        # data: (S,T,F)
        sid = index // self._n_per_series
        t0 = index % self._n_per_series

        series = self._data[sid]
        history_data = series[t0: t0 + self.input_len]
        future_data = series[t0 + self.input_len: t0 + self.input_len + self.output_len]
        # # ✅ STGCN 需要 [T, N, C]
        # history_data = history_data[..., np.newaxis]
        # future_data = future_data[..., np.newaxis]
        item["inputs"] = history_data.copy() if self.memmap else history_data
        item["targets"] = future_data.copy() if self.memmap else future_data

        if self.use_timestamps:
            ts_series = self.timestamps[sid]
            history_ts = ts_series[t0: t0 + self.input_len]
            future_ts = ts_series[t0 + self.input_len: t0 + self.input_len + self.output_len]
            item["inputs_timestamps"] = history_ts.copy() if self.memmap else history_ts
            item["targets_timestamps"] = future_ts.copy() if self.memmap else future_ts
        # print("input shape:", item["inputs"].shape)
        # print("target shape:", item["targets"].shape)
        return item

    def __len__(self) -> int:
        if not self._multi_series:
            return len(self._data) - self.input_len - self.output_len + 1
        return self._num_series * self._n_per_series

    @property
    def data(self) -> np.ndarray:
        return self._data
