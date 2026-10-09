"""Load split training trajectories and fit ROM parameters across trajectories."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from models.rom.spm import SPMModel
from scripts.rom.inference_common import _read_manifest, _read_sequence


def load_trajectories(
    data_dir: Path,
    manifest_path: Path,
    sequences_per_condition: int,
    max_points_per_sequence: int,
    seed: int,
) -> tuple[list[pd.DataFrame], float]:
    """Select independent, contiguous windows from the training split."""
    if sequences_per_condition <= 0 or max_points_per_sequence < 2:
        raise ValueError("轨迹数必须为正，且每条轨迹至少保留 2 个采样点")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"找不到训练数据目录: {data_dir}")

    manifest = _read_manifest(manifest_path, expected_split="train")
    generator = np.random.default_rng(seed)
    trajectories = []
    intervals = []
    for _, group in manifest.groupby("condition_code", sort=True):
        group = group.sort_values("sequence_file")
        selected = generator.choice(
            len(group), size=min(sequences_per_condition, len(group)), replace=False
        )
        for index in selected:
            sequence_file = group.iloc[int(index)]["sequence_file"]
            frame = _read_sequence(data_dir / sequence_file)
            if len(frame) < 2:
                raise ValueError(f"训练轨迹少于 2 个采样点: {sequence_file}")
            if len(frame) > max_points_per_sequence:
                start = int(generator.integers(0, len(frame) - max_points_per_sequence + 1))
                frame = frame.iloc[start:start + max_points_per_sequence].reset_index(drop=True)
            seconds = pd.to_datetime(frame["acqtime"], errors="raise")
            diffs = seconds.diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)
            if not np.isfinite(diffs).all() or (diffs <= 0).any():
                raise ValueError(f"训练轨迹时间戳无效: {sequence_file}")
            intervals.append(diffs)
            trajectories.append(frame)

    if not trajectories:
        raise ValueError(f"训练清单没有轨迹: {manifest_path}")
    return trajectories, float(np.median(np.concatenate(intervals)))


def estimate_equivalent_capacity(trajectories: list[pd.DataFrame]) -> float:
    """Fit the capacity in the models' SOC = SOC - rate * dt / (3600 * C) rule."""
    sum_xy = 0.0
    sum_xx = 0.0
    for frame in trajectories:
        timestamps = pd.to_datetime(frame["acqtime"])
        seconds = timestamps.diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)
        current = frame["rate"].to_numpy(dtype=float)[:-1]
        soc = frame["soc"].to_numpy(dtype=float)
        x = current * seconds / 3600.0
        y = soc[:-1] - soc[1:]
        sum_xy += float(np.dot(x, y))
        sum_xx += float(np.dot(x, x))
    if sum_xx == 0 or sum_xy <= 0:
        raise ValueError("训练轨迹无法估计正的等效容量；请检查 rate 与 SOC")
    return sum_xx / sum_xy


def _padded_arrays(trajectories: list[pd.DataFrame]):
    lengths = np.array([len(frame) for frame in trajectories])
    width = int(lengths.max())
    shape = (len(trajectories), width)
    soc = np.zeros(shape)
    voltage = np.zeros(shape)
    current = np.zeros(shape)
    mask = np.arange(width)[None, :] < lengths[:, None]
    for index, frame in enumerate(trajectories):
        count = lengths[index]
        soc[index, :count] = frame["soc"].to_numpy(dtype=float)
        voltage[index, :count] = frame["voltage"].to_numpy(dtype=float)
        current[index, :count] = frame["rate"].to_numpy(dtype=float)
        soc[index, count:] = soc[index, count - 1]
    return soc, voltage, current, mask


def _result_dict(result) -> dict:
    return {
        "params": np.asarray(result.x, dtype=float),
        "success": bool(result.success),
        "message": str(result.message),
        "mse": float(np.mean(result.fun ** 2)),
        "nfev": int(result.nfev),
    }


def calibrate_pngv(
    trajectories: list[pd.DataFrame], capacity: float, dt: float, max_nfev: int
) -> dict:
    """Fit voltage parameters while keeping capacity identified from SOC changes."""
    soc, voltage, current, mask = _padded_arrays(trajectories)
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("等效容量必须为正且有限")
    initial = np.array([3.16112983, 0.43467053, -0.43786890, 0.17775586,
                        0.05390640, 0.07558190, 2998.06703037])
    lower = np.array([2.5, -2.0, -2.0, -2.0, 1e-4, 1e-4, 100.0])
    upper = np.array([4.5, 2.0, 2.0, 2.0, 0.2, 0.2, 10000.0])

    def residuals(params):
        a0, a1, a2, a3, r0, rp, cp = params
        alpha = np.exp(-dt / (rp * cp))
        vp = np.zeros(len(trajectories))
        predicted = np.empty_like(voltage)
        for step in range(voltage.shape[1]):
            s = soc[:, step]
            predicted[:, step] = a0 + s * (a1 + s * (a2 + s * a3)) - r0 * current[:, step] - vp
            vp = alpha * vp + rp * (1 - alpha) * current[:, step]
        return (predicted - voltage)[mask]

    result = least_squares(residuals, initial, bounds=(lower, upper), max_nfev=max_nfev)
    output = _result_dict(result)
    output["params"] = np.r_[output["params"], capacity]
    return output


def calibrate_spm(trajectories: list[pd.DataFrame], dt: float, max_nfev: int) -> dict:
    """Fit the SPM voltage equation while resetting its state for each trajectory."""
    soc, voltage, current, mask = _padded_arrays(trajectories)
    model = SPMModel(dt=dt)
    physical = model.p
    initial = np.array([
        1.86170835, -1.88746860, 1.20236145, -0.79898473,
        2.41955830, 4.99999986, -4.27435781, 1.93755938,
        82.10038182, 811.05382216, 9.99999994e-4, 9.39979150e-4,
        0.07852909,
    ])
    lower = np.array([-1., -5., -5., -5., 2., -5., -5., -5.,
                      1., 1., 1e-7, 1e-7, 1e-4])
    upper = np.array([2., 5., 5., 5., 5., 5., 5., 5.,
                      1000., 1000., 1e-3, 1e-3, 0.2])
    theta_n_initial, theta_p_initial = model._soc_to_theta(soc[:, 0])
    eta_scale = 2.0 * physical.Rg * physical.T / physical.F

    def residuals(params):
        an = params[:4]
        ap = params[4:8]
        tau_n, tau_p, kn, kp, r0 = params[8:]
        csn = theta_n_initial * physical.csn_max
        csp = theta_p_initial * physical.csp_max
        predicted = np.empty_like(voltage)
        for step in range(voltage.shape[1]):
            tn = np.clip(csn / physical.csn_max, 0.0, 1.0)
            tp = np.clip(csp / physical.csp_max, 0.0, 1.0)
            un = an[0] + tn * (an[1] + tn * (an[2] + tn * an[3]))
            up = ap[0] + tp * (ap[1] + tp * (ap[2] + tp * ap[3]))
            i0n = kn * np.sqrt(np.clip(tn * (1.0 - tn), 1e-8, None))
            i0p = kp * np.sqrt(np.clip(tp * (1.0 - tp), 1e-8, None))
            rate = current[:, step]
            eta_n = eta_scale * np.arcsinh(rate / (2.0 * physical.Sn * i0n + 1e-12))
            eta_p = eta_scale * np.arcsinh(rate / (2.0 * physical.Sp * i0p + 1e-12))
            predicted[:, step] = up - un + eta_p - eta_n - rate * r0
            if step + 1 < voltage.shape[1]:
                next_soc = np.clip(soc[:, step + 1], 0.0, 1.0)
                theta_n, theta_p = model._soc_to_theta(next_soc)
                csn += dt / tau_n * (theta_n * physical.csn_max - csn) - dt * rate * 5.0
                csp += dt / tau_p * (theta_p * physical.csp_max - csp) + dt * rate * 5.0
                csn = np.clip(csn, 1.0, physical.csn_max - 1.0)
                csp = np.clip(csp, 1.0, physical.csp_max - 1.0)
        return (predicted - voltage)[mask]

    result = least_squares(residuals, initial, bounds=(lower, upper), max_nfev=max_nfev)
    return _result_dict(result)
