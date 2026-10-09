import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from typing import Tuple, Dict
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SPMParams:
    def __init__(self):
        self.F = 96485.0
        self.Rg = 8.314
        self.T = 298.15

        self.capacity_ah = 1.0
        self.ce = 1000.0

        # 最大固相浓度
        self.csn_max = 3.1e4
        self.csp_max = 5.1e4

        # 简化后的 lumped reaction scale
        self.Sn = 1.0
        self.Sp = 1.0

        # SOC -> stoichiometry 线性映射
        self.theta_n_0 = 0.02
        self.theta_n_100 = 0.85
        self.theta_p_0 = 0.95
        self.theta_p_100 = 0.45


class SPMModel:
    """
    参数结构:
    [an0, an1, an2, an3, ap0, ap1, ap2, ap3, tau_n, tau_p, kn, kp, R0]

    状态:
    [soc, csn_surf, csp_surf]
    """

    def __init__(self, dt: float = 1.0, spm_params: SPMParams = None):
        self.model_type = 'SPM'
        self.params = None
        self.dt = dt
        self.p = spm_params if spm_params is not None else SPMParams()
        self.state = None

    def reset_state(self, initial_soc: float) -> np.ndarray:
        self.state = self._init_state_from_soc(initial_soc)
        return self.state.copy()

    def set_state(self, state: np.ndarray):
        self.state = np.array(state, dtype=float).copy()

    def get_state(self) -> np.ndarray:
        if self.state is None:
            return None
        return self.state.copy()

    def _poly_ocv(self, theta: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        theta = np.clip(theta, 0.0, 1.0)
        a0, a1, a2, a3 = coeffs
        return a0 + a1 * theta + a2 * theta ** 2 + a3 * theta ** 3

    def _soc_to_theta(self, soc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        soc = np.clip(soc, 0.0, 1.0)
        theta_n = self.p.theta_n_0 + soc * (self.p.theta_n_100 - self.p.theta_n_0)
        theta_p = self.p.theta_p_0 + soc * (self.p.theta_p_100 - self.p.theta_p_0)
        return theta_n, theta_p

    def _theta_to_concentration(self, theta_n: float, theta_p: float) -> Tuple[float, float]:
        csn = theta_n * self.p.csn_max
        csp = theta_p * self.p.csp_max
        return csn, csp

    def _init_state_from_soc(self, soc: float) -> np.ndarray:
        theta_n, theta_p = self._soc_to_theta(soc)
        csn_avg, csp_avg = self._theta_to_concentration(theta_n, theta_p)
        csn_surf = csn_avg
        csp_surf = csp_avg
        return np.array([soc, csn_surf, csp_surf], dtype=float)

    def _state_transition(self, state: np.ndarray, current: float, params: np.ndarray) -> np.ndarray:
        soc, csn_surf, csp_surf = state
        _, _, _, _, _, _, _, _, tau_n, tau_p, _, _, _ = params

        # SOC 递推
        soc_next = soc - (current * self.dt) / (3600 * self.p.capacity_ah)
        soc_next = np.clip(soc_next, 0.0, 1.0)

        # 平均浓度由 SOC 决定
        theta_n_avg, theta_p_avg = self._soc_to_theta(soc_next)
        csn_avg, csp_avg = self._theta_to_concentration(theta_n_avg, theta_p_avg)

        # 表面浓度动态
        csn_surf_next = csn_surf + self.dt / tau_n * (csn_avg - csn_surf) - self.dt * current * 5.0
        csp_surf_next = csp_surf + self.dt / tau_p * (csp_avg - csp_surf) + self.dt * current * 5.0

        csn_surf_next = np.clip(csn_surf_next, 1.0, self.p.csn_max - 1.0)
        csp_surf_next = np.clip(csp_surf_next, 1.0, self.p.csp_max - 1.0)

        return np.array([soc_next, csn_surf_next, csp_surf_next], dtype=float)

    def _measurement(self, state: np.ndarray, current: float, params: np.ndarray) -> float:
        _, csn_surf, csp_surf = state
        an0, an1, an2, an3, ap0, ap1, ap2, ap3, _, _, kn, kp, R0 = params

        neg_coeffs = np.array([an0, an1, an2, an3], dtype=float)
        pos_coeffs = np.array([ap0, ap1, ap2, ap3], dtype=float)

        theta_n_surf = np.clip(csn_surf / self.p.csn_max, 0.0, 1.0)
        theta_p_surf = np.clip(csp_surf / self.p.csp_max, 0.0, 1.0)

        Un = self._poly_ocv(theta_n_surf, neg_coeffs)
        Up = self._poly_ocv(theta_p_surf, pos_coeffs)

        i0n = kn * np.sqrt(np.clip(theta_n_surf * (1.0 - theta_n_surf), 1e-8, None))
        i0p = kp * np.sqrt(np.clip(theta_p_surf * (1.0 - theta_p_surf), 1e-8, None))

        eta_n = (2.0 * self.p.Rg * self.p.T / self.p.F) * np.arcsinh(
            current / (2.0 * self.p.Sn * i0n + 1e-12)
        )
        eta_p = (2.0 * self.p.Rg * self.p.T / self.p.F) * np.arcsinh(
            current / (2.0 * self.p.Sp * i0p + 1e-12)
        )

        voltage = Up - Un + eta_p - eta_n - current * R0
        return float(voltage)

    def _extract_internal_variables(self, state: np.ndarray) -> Dict[str, float]:
        soc, csn_surf, csp_surf = state

        theta_n_avg, theta_p_avg = self._soc_to_theta(soc)
        theta_n_surf = np.clip(csn_surf / self.p.csn_max, 0.0, 1.0)
        theta_p_surf = np.clip(csp_surf / self.p.csp_max, 0.0, 1.0)

        return {
            'soc': float(soc),
            'theta_n_avg': float(theta_n_avg),
            'theta_p_avg': float(theta_p_avg),
            'theta_n_surf': float(theta_n_surf),
            'theta_p_surf': float(theta_p_surf)
        }

    def _simulate_sequence(self, soc: np.ndarray, current: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        用于拟合：
        使用真实 SOC 轨迹作为已知输入，仅递推表面浓度动态并拟合电压。
        这样比完全闭环更稳健。
        """
        n = len(current)
        voltage_pred = np.zeros(n)

        # 初始状态
        state = self._init_state_from_soc(soc[0])

        for i in range(n):
            # 当前电压
            voltage_pred[i] = self._measurement(state, current[i], params)

            if i < n - 1:
                _, csn_surf, csp_surf = state
                _, _, _, _, _, _, _, _, tau_n, tau_p, _, _, _ = params

                soc_next = np.clip(soc[i + 1], 0.0, 1.0)
                theta_n_avg, theta_p_avg = self._soc_to_theta(soc_next)
                csn_avg, csp_avg = self._theta_to_concentration(theta_n_avg, theta_p_avg)

                csn_surf_next = csn_surf + self.dt / tau_n * (csn_avg - csn_surf) - self.dt * current[i] * 5.0
                csp_surf_next = csp_surf + self.dt / tau_p * (csp_avg - csp_surf) + self.dt * current[i] * 5.0

                csn_surf_next = np.clip(csn_surf_next, 1.0, self.p.csn_max - 1.0)
                csp_surf_next = np.clip(csp_surf_next, 1.0, self.p.csp_max - 1.0)

                state = np.array([soc_next, csn_surf_next, csp_surf_next], dtype=float)

        return voltage_pred

    def _residuals(self, params: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        _, _, _, _, _, _, _, _, tau_n, tau_p, kn, kp, R0 = params

        if tau_n <= 0 or tau_p <= 0 or kn <= 0 or kp <= 0 or R0 <= 0:
            return 1e6 * np.ones(len(data))

        soc = data['soc'].values
        voltage = data['voltage'].values
        current = data['rate'].values

        voltage_pred = self._simulate_sequence(soc, current, params)

        if np.any(np.isnan(voltage_pred)) or np.any(np.isinf(voltage_pred)):
            return 1e6 * np.ones(len(data))

        return voltage_pred - voltage

    def fit(self, data: pd.DataFrame, initial_params: np.ndarray = None,
            max_nfev: int = 500) -> Dict:
        data = data.copy()
        if data['soc'].max() > 1.5:
            data['soc'] = data['soc'] / 100.0
        if 'acqtime' in data.columns:
            sample_intervals = pd.to_datetime(data['acqtime']).diff().dt.total_seconds()
            sample_intervals = sample_intervals[sample_intervals > 0]
            if not sample_intervals.empty:
                self.dt = float(sample_intervals.median())

        if initial_params is None:
            initial_params = np.array([
                1.86170835, -1.88746860, 1.20236145, -0.79898473,
                2.41955830, 4.99999986, -4.27435781, 1.93755938,
                82.10038182, 811.05382216,
                9.99999994e-4, 9.39979150e-4,
                0.07852909
            ], dtype=float)

        lower_bounds = np.array([
            -1.0, -5.0, -5.0, -5.0,
             2.0, -5.0, -5.0, -5.0,
             1.0, 1.0,
             1e-7, 1e-7,
             1e-4
        ], dtype=float)

        upper_bounds = np.array([
             2.0, 5.0, 5.0, 5.0,
             5.0, 5.0, 5.0, 5.0,
             1000.0, 1000.0,
             1e-3, 1e-3,
             0.2
        ], dtype=float)

        result = least_squares(
            fun=self._residuals,
            x0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            args=(data,),
            max_nfev=max_nfev,
            verbose=0
        )

        self.params = result.x
        mse = np.mean(result.fun ** 2)

        return {
            'params': result.x,
            'success': result.success,
            'message': result.message,
            'mse': mse
        }

    def predict_step(self, current_soc: float = None, current_voltage: float = None,
                     current: float = 0.0, params: np.ndarray = None,
                     return_internal: bool = False) -> Dict:
        """
        更严格状态空间形式：
        - 若 self.state 已存在，则直接基于内部状态递推
        - 若 self.state 不存在，则必须提供 current_soc 用于初始化

        current_voltage 保留仅用于接口统一，不参与当前简化 SPM 状态更新
        """
        if params is None:
            if self.params is None:
                raise ValueError("请先训练模型或提供参数")
            params = self.params

        if self.state is None:
            if current_soc is None:
                raise ValueError("模型状态未初始化，请提供 current_soc 或先调用 reset_state()")
            self.reset_state(current_soc)
        else:
            if current_soc is not None:
                # 可选：如果用户传入 current_soc，则以该 SOC 重置状态
                self.reset_state(current_soc)

        next_state = self._state_transition(self.state, current, params)
        next_voltage = self._measurement(next_state, current, params)

        self.state = next_state.copy()

        result = {
            'next_soc': float(next_state[0]),
            'next_voltage': float(next_voltage)
        }

        if return_internal:
            internal_vars = self._extract_internal_variables(next_state)
            result.update({
                'theta_n_avg': internal_vars['theta_n_avg'],
                'theta_p_avg': internal_vars['theta_p_avg'],
                'theta_n_surf': internal_vars['theta_n_surf'],
                'theta_p_surf': internal_vars['theta_p_surf']
            })

        return result

    def predict_sequence(self, initial_soc: float, initial_voltage: float,
                         current_sequence: np.ndarray,
                         return_internal: bool = False,
                         soc_ground_truth: np.ndarray = None,
                         teacher_forcing_interval: int = None) -> Dict[str, np.ndarray]:
        if self.params is None:
            raise ValueError("请先训练模型")

        n_steps = len(current_sequence)
        if teacher_forcing_interval is not None:
            if teacher_forcing_interval <= 0:
                raise ValueError("teacher_forcing_interval 必须为正整数")
            if soc_ground_truth is None or len(soc_ground_truth) != n_steps:
                raise ValueError("teacher forcing 需要与电流序列等长的 SOC 真值")

        soc_pred = np.zeros(n_steps)
        voltage_pred = np.zeros(n_steps)

        if return_internal:
            theta_n_avg_pred = np.zeros(n_steps)
            theta_p_avg_pred = np.zeros(n_steps)
            theta_n_surf_pred = np.zeros(n_steps)
            theta_p_surf_pred = np.zeros(n_steps)

        self.reset_state(initial_soc)

        # 初值
        soc_pred[0] = initial_soc
        voltage_pred[0] = initial_voltage

        if return_internal:
            init_vars = self._extract_internal_variables(self.state)
            theta_n_avg_pred[0] = init_vars['theta_n_avg']
            theta_p_avg_pred[0] = init_vars['theta_p_avg']
            theta_n_surf_pred[0] = init_vars['theta_n_surf']
            theta_p_surf_pred[0] = init_vars['theta_p_surf']

        # 严格状态空间递推
        for i in range(n_steps - 1):
            next_state = self._state_transition(self.state, current_sequence[i], self.params)
            next_voltage = self._measurement(next_state, current_sequence[i + 1], self.params)

            self.state = next_state.copy()

            soc_pred[i + 1] = next_state[0]
            voltage_pred[i + 1] = next_voltage

            if return_internal:
                internal_vars = self._extract_internal_variables(next_state)
                theta_n_avg_pred[i + 1] = internal_vars['theta_n_avg']
                theta_p_avg_pred[i + 1] = internal_vars['theta_p_avg']
                theta_n_surf_pred[i + 1] = internal_vars['theta_n_surf']
                theta_p_surf_pred[i + 1] = internal_vars['theta_p_surf']

            if (teacher_forcing_interval is not None
                    and (i + 1) % teacher_forcing_interval == 0):
                self.reset_state(float(soc_ground_truth[i + 1]))

        result = {
            'soc': soc_pred,
            'voltage': voltage_pred
        }

        if return_internal:
            result.update({
                'theta_n_avg': theta_n_avg_pred,
                'theta_p_avg': theta_p_avg_pred,
                'theta_n_surf': theta_n_surf_pred,
                'theta_p_surf': theta_p_surf_pred
            })

        return result

    def save_checkpoint(self, path, metadata: Dict = None) -> None:
        if self.params is None:
            raise ValueError("模型参数为空，无法保存权重")

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint_path,
            model_type=self.model_type,
            params=np.asarray(self.params, dtype=float),
            dt=float(self.dt),
            capacity_ah=float(self.p.capacity_ah),
            parameter_names=np.array([
                'an0', 'an1', 'an2', 'an3', 'ap0', 'ap1', 'ap2', 'ap3',
                'tau_n', 'tau_p', 'kn', 'kp', 'R0'
            ]),
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False)
        )

    @classmethod
    def load_checkpoint(cls, path, spm_params: SPMParams = None):
        with np.load(path, allow_pickle=False) as checkpoint:
            model_type = str(checkpoint['model_type'].item())
            if model_type != 'SPM':
                raise ValueError(f"权重类型不匹配: 期望 SPM，实际为 {model_type}")
            model = cls(
                dt=float(checkpoint['dt'].item()),
                spm_params=spm_params
            )
            if 'capacity_ah' in checkpoint.files:
                model.p.capacity_ah = float(checkpoint['capacity_ah'].item())
            model.params = checkpoint['params'].astype(float)
        return model


def generate_synthetic_data(noise_level: float = 0.003,
                            n_points: int = 100,
                            dt: float = 1.0) -> Tuple[pd.DataFrame, np.ndarray]:
    p = SPMParams()

    true_params = np.array([
        0.9, -0.8, 0.3, -0.1,
        4.1, 0.6, -0.2, 0.1,
        80.0, 120.0,
        2e-5, 2.5e-5,
        0.025
    ], dtype=float)

    time = np.arange(n_points) * dt

    current = np.zeros(n_points)
    for k in range(0, n_points, 50):
        amp = np.random.choice([0.0, 0.5, 1.0, 1.5, 2.0])
        current[k:k + 25] = amp

    soc = np.zeros(n_points)
    soc[0] = 0.95
    for k in range(1, n_points):
        soc[k] = soc[k - 1] - current[k - 1] * dt / (3600 * p.capacity_ah)
    soc = np.clip(soc, 0.0, 1.0)

    model = SPMModel(dt=dt, spm_params=p)
    voltage = model._simulate_sequence(soc, current, true_params)
    voltage += np.random.normal(0, noise_level, n_points)

    data = pd.DataFrame({
        'time': time,
        'voltage': voltage,
        'soc': soc,
        'rate': current
    })

    return data, true_params


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = ['time', 'voltage', 'soc', 'rate']

    file_name_train = '../Dataset/NCM_2_0/spme_dcc_0.4s_298k'
    file_name_test = '../Dataset/NCM_2_0/spme_dcc_0.6s_298k'

    df_train = pd.read_csv(f"{file_name_train}.csv", usecols=columns)
    df_test = pd.read_csv(f"{file_name_test}.csv", usecols=columns)

    data_train = df_train[columns]
    data_test = df_test[columns]

    return data_train, data_test


if __name__ == "__main__":
    print("读取数据...")
    # 若没有真实数据，可改为：
    data_train, true_params = generate_synthetic_data()
    data_test, _ = generate_synthetic_data()

    # data_train, data_test = get_data()

    time = data_test['time'].values
    current = data_test['rate'].values
    true_soc = data_test['soc'].values
    true_voltage = data_test['voltage'].values

    model_spm = SPMModel(dt=1.0)

    print("训练SPM模型...")
    result_spm = model_spm.fit(data_train)
    print(f"SPM模型训练完成，MSE: {result_spm['mse']:.6f}")

    print("\n进行序列预测...")
    n_pred = min(1000, len(data_test))
    test_current = current[:n_pred]

    pred_result = model_spm.predict_sequence(
        initial_soc=true_soc[0],
        initial_voltage=true_voltage[0],
        current_sequence=test_current,
        return_internal=True
    )

    soc_pred_spm = pred_result['soc']
    voltage_pred_spm = pred_result['voltage']

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    time_pred = time[:n_pred]

    axes[0].plot(time_pred, true_voltage[:n_pred], 'b-', linewidth=2, label='真实电压')
    axes[0].plot(time_pred, voltage_pred_spm, 'r--', linewidth=1.5, label='SPM预测电压')
    axes[0].set_xlabel('时间 (秒)', fontsize=12)
    axes[0].set_ylabel('电压 (V)', fontsize=12)
    axes[0].set_title('SPM电池电压预测结果对比', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_pred, true_soc[:n_pred], 'b-', linewidth=2, label='真实SOC')
    axes[1].plot(time_pred, soc_pred_spm, 'r--', linewidth=1.5, label='SPM预测SOC')
    axes[1].set_xlabel('时间 (秒)', fontsize=12)
    axes[1].set_ylabel('SOC', fontsize=12)
    axes[1].set_title('SPM电池SOC预测结果对比', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    if 'theta_n_avg' in pred_result:
        axes[2].plot(time_pred, pred_result['theta_n_avg'], label='theta_n_avg')
        axes[2].plot(time_pred, pred_result['theta_p_avg'], label='theta_p_avg')
        axes[2].plot(time_pred, pred_result['theta_n_surf'], '--', label='theta_n_surf')
        axes[2].plot(time_pred, pred_result['theta_p_surf'], '--', label='theta_p_surf')
        axes[2].set_xlabel('时间 (秒)', fontsize=12)
        axes[2].set_ylabel('Theta', fontsize=12)
        axes[2].set_title('SPM内部化学计量比预测结果', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    voltage_error_spm = np.mean(np.abs(voltage_pred_spm - true_voltage[:n_pred]))
    soc_error_spm = np.mean(np.abs(soc_pred_spm - true_soc[:n_pred]))

    print("\n预测误差统计:")
    print(f"SPM模型 - 电压平均绝对误差: {voltage_error_spm:.6f} V")
    print(f"SPM模型 - SOC平均绝对误差: {soc_error_spm:.6f}")

    print("\nSPM模型参数:")
    param_names = [
        "an0", "an1", "an2", "an3",
        "ap0", "ap1", "ap2", "ap3",
        "tau_n", "tau_p", "kn", "kp", "R0"
    ]
    for name, value in zip(param_names, model_spm.params):
        print(f"{name}: {value:.6e}")
