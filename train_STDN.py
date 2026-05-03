from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.metrics import masked_mse
from basicts.runners.callback import EarlyStopping, GradientClipping

from dataset.hpibt_dataset import HybridPIBTDataset
from dataset.graph_scaler import GraphScaler
from model.STDN import STDN, STDNConfig

import numpy as np


def main():
    data = np.load("data/den520d_lifelong/robot1000/train_data.npy")
    adj = np.load("data/den520d_adjacency.npy")
    print(data.shape)
    print(data[:1])

    num_nodes = data.shape[-1]

    for input_len in [12, 24, 48, 96]:
        for output_len in [12, 24, 48, 96]:
            if output_len > input_len:
                continue

            model_config = STDNConfig(
                input_len=input_len,
                output_len=output_len,
                num_nodes=num_nodes,
                time_of_day_size=288,
                bn_decay=0.1,
                L=2,
                K=16,
                d=8,
                reference=3,
                order=3,
                in_channels=1,
                out_channels=1,
                node_miss_rate=0.1,
                T_miss_len=12,
                adj=adj.tolist(),
            )

            BasicTSLauncher.launch_training(BasicTSForecastingConfig(
                model=STDN,
                model_config=model_config,
                scaler=GraphScaler,
                norm_each_channel=True,
                dataset_name="traffic_flow",
                dataset_type=HybridPIBTDataset,
                data_file_path="data/den520d_lifelong/robot1000",
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
                optimizer_params={"lr": 5e-4},
                lr_scheduler=MultiStepLR,
                lr_scheduler_params={"milestones": [25, 50], "gamma": 0.5},
            ))


if __name__ == "__main__":
    main()
