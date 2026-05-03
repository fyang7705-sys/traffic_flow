import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.metrics import masked_mse
from basicts.runners.callback import EarlyStopping, GradientClipping

from dataset.graph_scaler import GraphScaler
from dataset.hpibt_dataset import HybridPIBTDataset
from model.HimNet import HimNet, HimNetConfig


def main():
    data = np.load("data/den520d_lifelong/robot1000/train_data.npy")
    num_nodes = data.shape[-1]
    for input_len in [12, 24, 48, 96]:
        for output_len in [12, 24, 48, 96]:
            if output_len > input_len:
                continue
            model_config = HimNetConfig(
                input_len=input_len, 
                output_len=output_len, 
                num_nodes=num_nodes,
                use_time_embedding=False
            )
            BasicTSLauncher.launch_training(BasicTSForecastingConfig(
                model=HimNet, 
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
                metrics=["MAE", "MSE", "RMSE", "MAPE", "WAPE"], loss=masked_mse,
                optimizer_params={"lr": 5e-4}, lr_scheduler=MultiStepLR,
                lr_scheduler_params={"milestones": [25, 50], "gamma": 0.5},
            ))


if __name__ == "__main__":
    main()
