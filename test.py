import os
import json
import glob
from pathlib import Path

import numpy as np
import torch

from dataset.graph_scaler import GraphScaler


def to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def calc_metrics(prediction: torch.Tensor, targets: torch.Tensor, eps: float = 1e-5):
    assert prediction.shape == targets.shape, f"shape mismatch: {prediction.shape} vs {targets.shape}"

    error = prediction - targets
    abs_error = torch.abs(error)

    mae = torch.mean(abs_error)
    mse = torch.mean(error ** 2)
    rmse = torch.sqrt(mse)

    valid_mask = torch.abs(targets) > eps
    if valid_mask.any():
        mape = torch.mean(abs_error[valid_mask] / torch.abs(targets[valid_mask])) * 100
    else:
        mape = torch.tensor(float("nan"))

    wmape = torch.sum(abs_error) / torch.sum(torch.abs(targets).clamp_min(eps)) * 100

    return {
        "mae": to_float(mae),
        "mse": to_float(mse),
        "rmse": to_float(rmse),
        "mape": to_float(mape),
        "wmape": to_float(wmape),
    }


def calc_horizon_metrics(prediction: torch.Tensor, targets: torch.Tensor, eps: float = 1e-5):
    horizon_results = {}

    for h in range(prediction.shape[1]):
        pred_h = prediction[:, h]
        target_h = targets[:, h]

        horizon_results[f"horizon_{h + 1}"] = calc_metrics(pred_h, target_h, eps=eps)

    return horizon_results


def load_memmap_tensor(path: str, total_samples: int, input_shape, dtype):
    arr = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        shape=(total_samples, *input_shape),
    )

    # memmap mode="r" 是只读的，torch.from_numpy 前 copy 避免 warning
    return torch.from_numpy(arr.copy()).float()


def main():
    checkpoints_dir = "./checkpoints"
    train_data_path = "../../data/traffic_flow/den520d_lifelong/robot1000/train_data.npy"

    input_shape = (12, 202, 1)
    dtype = np.float32
    eps = 1e-5

    scaler = GraphScaler(norm_each_channel=True)
    scaler.fit(np.load(train_data_path))

    # 修正 scaler mean/std shape: [1, 1, 202] -> [1, 1, 202, 1]
    if scaler.stats["mean"].ndim == 3:
        scaler.stats["mean"] = scaler.stats["mean"].unsqueeze(-1)
        scaler.stats["std"] = scaler.stats["std"].unsqueeze(-1)

    pred_paths = glob.glob(
        os.path.join(
            checkpoints_dir,
            "*",
            "traffic_flow_*",
            "*",
            "test_results",
            "prediction.npy",
        )
    )

    pred_paths = sorted(pred_paths)

    print(f"Found {len(pred_paths)} prediction files.")

    single_sample_size = np.prod(input_shape) * np.dtype(dtype).itemsize

    for pred_path in pred_paths:
        test_results_dir = Path(pred_path).parent
        hash_dir = test_results_dir.parent

        targets_path = test_results_dir / "targets.npy"
        results_path = hash_dir / "results.json"
        if results_path.exists():
            print(f"[SKIP] results.json already exists: {results_path}")
            continue
        
        if not targets_path.exists():
            print(f"[SKIP] targets.npy not found: {targets_path}")
            continue

        pred_file_size = os.path.getsize(pred_path)
        targets_file_size = os.path.getsize(targets_path)

        if pred_file_size != targets_file_size:
            print(f"[SKIP] file size mismatch:")
            print(f"       prediction: {pred_path}, size={pred_file_size}")
            print(f"       targets:    {targets_path}, size={targets_file_size}")
            continue

        if pred_file_size % single_sample_size != 0:
            print(f"[SKIP] invalid file size: {pred_path}")
            continue

        total_samples = pred_file_size // single_sample_size

        print(f"\nProcessing: {hash_dir}")
        print(f"total_samples: {total_samples}")

        prediction = load_memmap_tensor(
            pred_path,
            total_samples=total_samples,
            input_shape=input_shape,
            dtype=dtype,
        )

        targets = load_memmap_tensor(
            str(targets_path),
            total_samples=total_samples,
            input_shape=input_shape,
            dtype=dtype,
        )

        prediction = scaler.inverse_transform(prediction)
        targets = scaler.inverse_transform(targets)

        print("prediction shape:", prediction.shape)
        print("targets shape:", targets.shape)

        overall_metrics = calc_metrics(prediction, targets, eps=eps)
        horizon_metrics = calc_horizon_metrics(prediction, targets, eps=eps)

        result = {
            "model_name": hash_dir.parent.parent.name,
            "run_dir": str(hash_dir),
            "test_results_dir": str(test_results_dir),
            "prediction_path": str(pred_path),
            "targets_path": str(targets_path),
            "total_samples": int(total_samples),
            "input_shape": list(input_shape),
            "prediction_shape": list(prediction.shape),
            "targets_shape": list(targets.shape),
            "overall": overall_metrics,
            "horizon": horizon_metrics,
        }

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"[OK] saved: {results_path}")
        print(
            f"MAE={overall_metrics['mae']:.6f}, "
            f"MSE={overall_metrics['mse']:.6f}, "
            f"RMSE={overall_metrics['rmse']:.6f}, "
            f"MAPE={overall_metrics['mape']:.6f}%, "
            f"WMAPE={overall_metrics['wmape']:.6f}%"
        )


if __name__ == "__main__":
    main()
