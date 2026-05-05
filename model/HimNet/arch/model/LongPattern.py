import numpy as np


class LongPatternExtractor:
    @staticmethod
    def extract_long_pattern(data):
        """从训练集离线统计长期特征。

        Args:
            data: np.ndarray
                期望形状 [S, T, N, C]，本文场景 C=2（value + local_ends）

        Returns:
            long_feat: np.ndarray
                形状 [N, C]，对 S、T 两维做均值聚合得到的长期特征
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if data.ndim != 4:
            raise ValueError(f"Expected data shape [S,T,N,C], got {data.shape}")

        # [S,T,N,C] -> [N,C]
        long_feat = data.mean(axis=(0, 1))
        return long_feat
