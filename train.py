from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.metrics import masked_mse
from basicts.runners.callback import EarlyStopping, GradientClipping

# 使用自定义数据集：支持 (S,T,F)
from dataset.hpibt_dataset import HybridPIBTDataset

# 使用 STGCN
from model.STGCN import STGCN, STGCNConfig

import numpy as np


def main():

    data = np.load("data/den520d_lifelong/robot1000/train_data.npy")
    print(data.shape)
    print(data[:1])

    num_nodes = 202

    # 用单位矩阵作为占位邻接矩阵（num_nodes = num_features）
    adj = np.eye(num_nodes, dtype=np.float32)

    for input_len in [12, 24, 48, 96]:
        for output_len in [12, 24, 48, 96]:
            if output_len > input_len:
                continue

            model_config = STGCNConfig(
                num_nodes=num_nodes,
                in_channels=1,
                out_channels=1,
                input_len=input_len,
                output_len=output_len,
                hidden_channels=64,
                num_layers=2,
                kernel_size=3,
                dropout=0.0,
            )

            BasicTSLauncher.launch_training(BasicTSForecastingConfig(
                model=STGCN,
                model_config=model_config,
                # 通过额外参数把邻接矩阵传给模型的 __init__(config, adj=...)
                model_init_kwargs={"adj": adj},
                dataset_name="traffic_flow",
                dataset_type=HybridPIBTDataset,
                data_file_path="datasets/traffic_flow/den520d_lifelong/robot1000",
                use_timestamps=False,
                input_len=input_len,
                output_len=output_len,
                gpus="0",
                callbacks=[EarlyStopping(), GradientClipping(1.0)],
                seed=233,
                num_epochs=100,
                batch_size=32,
                metrics=["MAE", "MSE", "RMSE", "MAPE", "WAPE"],
                loss=masked_mse,
                optimizer_params={
                    "lr": 5e-4
                },
                lr_scheduler=MultiStepLR,
                lr_scheduler_params={
                    "milestones": [25, 50],
                    "gamma": 0.5
                }
            ))


if __name__ == "__main__":
    main()
