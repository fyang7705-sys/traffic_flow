from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.metrics import masked_mse
from basicts.runners.callback import EarlyStopping, GradientClipping

from dataset.hpibt_dataset import HybridPIBTDataset
from dataset.graph_scaler import GraphScaler
from model.STID import STID, STIDConfig

import numpy as np


def main():
    data = np.load("data/den520d_lifelong/robot1000/train_data.npy")
    print(data.shape)
    print(data[:1])

    num_features = data.shape[-1]

    for input_len in [12, 24, 48, 96]:
        for output_len in [12, 24, 48, 96]:
            if output_len > input_len:
                continue

            model_config = STIDConfig(
                input_len=input_len,
                output_len=output_len,
                num_features=num_features,
                input_hidden_size=32,
                intermediate_size=None,
                hidden_act="relu",
                num_layers=3,
                if_spatial=True,
                spatial_hidden_size=32,
                if_time_in_day=False,
                if_day_in_week=False,
                num_time_in_day=24,
                num_day_in_week=7,
                tid_hidden_size=32,
                diw_hidden_size=32,
            )

            BasicTSLauncher.launch_training(BasicTSForecastingConfig(
                model=STID,
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
