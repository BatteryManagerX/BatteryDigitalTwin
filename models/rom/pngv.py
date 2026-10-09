import numpy as np
import pandas as pd
import json
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PNGVModel:
    def __init__(self, dt: float = 1.0):
        self.model_type = 'PNGV'
        self.params = None
        self.dt = dt

    def _ocv_from_soc(self, soc: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        OCV-SOC 多项式关系
        params:
            [a0, a1, a2, a3, R0, Rp, Cp, capacity]
        """
        coeffs = params[:4]
        ocv = coeffs[0] + coeffs[1] * soc + coeffs[2] * soc ** 2 + coeffs[3] * soc ** 3
        return ocv

    def _state_transition_pngv(self, state: np.ndarray, current: float, params: np.ndarray) -> np.ndarray:
        """
        PNGV状态更新
        state = [soc, vp]
            soc: 荷电状态
            vp : 极化电压
        """
        soc, vp = state
        R0, Rp, Cp, capacity = params[4:8]

        soc_next = soc - (current * self.dt) / (3600 * capacity)

        tau = Rp * Cp
        alpha = np.exp(-self.dt / tau)
        vp_next = alpha * vp + Rp * (1 - alpha) * current

        return np.array([soc_next, vp_next])

    def _measurement_pngv(self, state: np.ndarray, current: float, params: np.ndarray) -> float:
        """
        端电压方程
        vt = OCV - I*R0 - Vp
        """
        soc, vp = state
        R0 = params[4]
        ocv = self._ocv_from_soc(soc, params)
        voltage = ocv - R0 * current - vp
        return voltage

    def predict_step(self, current_soc: float, current_voltage: float,
                     current: float, params: np.ndarray = None) -> Tuple[float, float]:
        """
        单步预测：
        根据当前SOC、当前端电压和电流，反推出当前极化电压vp，
        然后做一步状态推进，输出下一时刻SOC和端电压
        """
        if params is None:
            if self.params is None:
                raise ValueError("请先训练模型或提供参数")
            params = self.params

        ocv = self._ocv_from_soc(current_soc, params)
        R0 = params[4]

        # 由当前测得电压反推极化电压
        vp = ocv - current_voltage - R0 * current

        state = np.array([current_soc, vp])
        next_state = self._state_transition_pngv(state, current, params)
        next_voltage = self._measurement_pngv(next_state, current, params)

        next_soc = next_state[0]
        return next_soc, next_voltage

    def _get_current_column(self, data: pd.DataFrame) -> str:
        if 'rate' in data.columns:
            return 'rate'
        elif 'current' in data.columns:
            return 'current'
        else:
            raise ValueError("数据中必须包含 'rate' 或 'current' 列")

    def _objective_function(self, params: np.ndarray, data: pd.DataFrame) -> float:
        """
        目标函数：电压MSE
        用真实SOC参与拟合，递推极化电压vp
        """
        voltage_errors = []

        soc = data['soc'].values
        voltage = data['voltage'].values
        current_col = self._get_current_column(data)
        current = data[current_col].values

        # 参数简单约束，防止数值问题
        R0, Rp, Cp, capacity = params[4:8]
        if R0 <= 0 or Rp <= 0 or Cp <= 0 or capacity <= 0:
            return 1e6

        vp = 0.0
        for i in range(len(data) - 1):
            state = np.array([soc[i], vp])
            pred_voltage = self._measurement_pngv(state, current[i], params)
            voltage_errors.append(pred_voltage - voltage[i])

            state = self._state_transition_pngv(state, current[i], params)
            vp = state[1]

        mse = np.mean(np.array(voltage_errors) ** 2)
        return mse

    def fit(self, data: pd.DataFrame, initial_params: np.ndarray = None,
            maxiter: int = 1000) -> Dict:
        """
        参数拟合
        params:
            [a0, a1, a2, a3, R0, Rp, Cp, capacity]
        """
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
                3.16112983,    # a0
                0.43467053,    # a1
                -0.43786890,   # a2
                0.17775586,    # a3
                0.05390640,    # R0
                0.07558190,    # Rp
                2998.06703037, # Cp
                1.01195062     # C-rate 数据对应的等效容量
            ])

        bounds = [
            (2.5, 4.5),      # a0
            (-2.0, 2.0),     # a1
            (-2.0, 2.0),     # a2
            (-2.0, 2.0),     # a3
            (1e-4, 0.2),     # R0
            (1e-4, 0.2),     # Rp
            (100.0, 10000),  # Cp
            (0.25, 4.0)      # C-rate 数据对应的等效容量
        ]

        result = minimize(
            fun=self._objective_function,
            x0=initial_params,
            args=(data,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': False}
        )

        self.params = result.x
        return {
            'params': result.x,
            'success': result.success,
            'message': result.message,
            'mse': result.fun
        }

    def predict_sequence(self, initial_soc: float, initial_voltage: float,
                         current_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        多步递推预测
        """
        if self.params is None:
            raise ValueError("请先训练模型")

        n_steps = len(current_sequence)
        soc_pred = np.zeros(n_steps)
        voltage_pred = np.zeros(n_steps)

        soc_pred[0] = initial_soc
        voltage_pred[0] = initial_voltage

        if n_steps == 0:
            return soc_pred, voltage_pred

        ocv = self._ocv_from_soc(initial_soc, self.params)
        vp = ocv - initial_voltage - self.params[4] * current_sequence[0]
        state = np.array([initial_soc, vp], dtype=float)

        for i in range(n_steps - 1):
            state = self._state_transition_pngv(state, current_sequence[i], self.params)
            soc_pred[i + 1] = state[0]
            voltage_pred[i + 1] = self._measurement_pngv(
                state, current_sequence[i + 1], self.params
            )

        return soc_pred, voltage_pred

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
            parameter_names=np.array([
                'a0', 'a1', 'a2', 'a3', 'R0', 'Rp', 'Cp', 'capacity'
            ]),
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False)
        )

    @classmethod
    def load_checkpoint(cls, path):
        with np.load(path, allow_pickle=False) as checkpoint:
            model_type = str(checkpoint['model_type'].item())
            if model_type != 'PNGV':
                raise ValueError(f"权重类型不匹配: 期望 PNGV，实际为 {model_type}")
            model = cls(dt=float(checkpoint['dt'].item()))
            model.params = checkpoint['params'].astype(float)
        return model


def generate_synthetic_data(noise_level: float = 0.005) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成PNGV合成数据
    """
    true_params = np.array([
        3.0,    # a0
        0.8,    # a1
        -0.3,   # a2
        0.2,    # a3
        0.03,   # R0
        0.02,   # Rp
        2000.0, # Cp
        2.0     # capacity
    ])

    dt = 1.0
    n_points = 400
    time = np.arange(n_points) * dt

    # 构造脉冲电流
    np.random.seed(42)
    current = np.zeros(n_points)
    for k in range(0, n_points, 40):
        amp = np.random.choice([0.0, 0.5, 1.0, 1.5])
        current[k:k + 20] = amp

    # SOC积分
    soc = np.zeros(n_points)
    soc[0] = 0.95
    for k in range(1, n_points):
        soc[k] = soc[k - 1] - current[k - 1] * dt / (3600 * true_params[7])
    soc = np.clip(soc, 0.0, 1.0)

    model = PNGVModel()
    voltage = np.zeros(n_points)
    vp = 0.0

    for i in range(n_points):
        state = np.array([soc[i], vp])
        voltage[i] = model._measurement_pngv(state, current[i], true_params)
        state = model._state_transition_pngv(state, current[i], true_params)
        vp = state[1]

    voltage += np.random.normal(0, noise_level, n_points)

    data = pd.DataFrame({
        'time': time,
        'voltage': voltage,
        'soc': soc,
        'rate': current
    })

    return data, true_params


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取真实数据，格式与第一个文件保持一致
    """
    columns = ['time', 'voltage', 'soc', 'rate']

    file_name_train = '../Dataset/NCM_2_0/spme_dcc_0.4s_298k'
    file_name_test = '../Dataset/NCM_2_0/spme_dcc_0.6s_298k'

    df_train = pd.read_csv(f"{file_name_train}.csv", usecols=columns)
    df_test = pd.read_csv(f"{file_name_test}.csv", usecols=columns)

    data_train = df_train[columns]
    data_test = df_test[columns]

    return data_train, data_test


# 主程序
if __name__ == "__main__":
    print("准备数据...")

    # 方式1：使用真实数据
    try:
        data_train, data_test = get_data()
        print("已读取真实数据。")
    except Exception as e:
        print(f"读取真实数据失败，改用合成数据。原因: {e}")
        data_train, true_params = generate_synthetic_data()
        data_test, _ = generate_synthetic_data()

    time = data_test['time']
    current = data_test['rate'] if 'rate' in data_test.columns else data_test['current']
    true_soc = data_test['soc']
    true_voltage = data_test['voltage']

    # 训练PNGV模型
    model_pngv = PNGVModel()
    print("训练PNGV模型...")
    result_pngv = model_pngv.fit(data_train)
    print(f"PNGV模型训练完成，MSE: {result_pngv['mse']:.6f}")

    # 序列预测
    print("\n进行序列预测...")
    n_pred = min(1000, len(data_test))
    test_current = current[:n_pred].values

    soc_pred, voltage_pred = model_pngv.predict_sequence(
        true_soc.iloc[0],
        true_voltage.iloc[0],
        test_current
    )

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    time_pred = time[:n_pred].values

    # 电压对比图
    axes[0].plot(time_pred, true_voltage[:n_pred], 'b-', linewidth=2, label='真实电压')
    axes[0].plot(time_pred, voltage_pred, 'r--', linewidth=1.5, label='PNGV预测电压')
    axes[0].set_xlabel('时间 (秒)', fontsize=12)
    axes[0].set_ylabel('电压 (V)', fontsize=12)
    axes[0].set_title('PNGV电池电压预测结果对比', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([time_pred[0], time_pred[-1]])

    # SOC对比图
    axes[1].plot(time_pred, true_soc[:n_pred], 'b-', linewidth=2, label='真实SOC')
    axes[1].plot(time_pred, soc_pred, 'r--', linewidth=1.5, label='PNGV预测SOC')
    axes[1].set_xlabel('时间 (秒)', fontsize=12)
    axes[1].set_ylabel('SOC', fontsize=12)
    axes[1].set_title('PNGV电池SOC预测结果对比', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([time_pred[0], time_pred[-1]])

    plt.tight_layout()
    plt.show()

    # 误差统计
    voltage_error = np.mean(np.abs(voltage_pred - true_voltage[:n_pred].values))
    soc_error = np.mean(np.abs(soc_pred - true_soc[:n_pred].values))

    print("\n预测误差统计:")
    print(f"PNGV模型 - 电压平均绝对误差: {voltage_error:.6f} V")
    print(f"PNGV模型 - SOC平均绝对误差: {soc_error:.6f}")

    # 打印模型参数
    print("\nPNGV模型参数:")
    print(f"OCV系数: [{', '.join([f'{p:.6f}' for p in model_pngv.params[:4]])}]")
    print(f"R0: {model_pngv.params[4]:.6f} Ω")
    print(f"Rp: {model_pngv.params[5]:.6f} Ω")
    print(f"Cp: {model_pngv.params[6]:.2f} F")
    print(f"容量: {model_pngv.params[7]:.2f} Ah")
