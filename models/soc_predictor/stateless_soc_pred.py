import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class ResidualCausalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.05):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(self.act(self.norm1(x)))
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(self.act(self.norm2(x)))
        return residual + x


class StatelessSoCPredictor(nn.Module):
    """Predict a complete SOC trajectory with shared trajectory parameters.

    ``rate_seq`` is expected to contain current in amperes, matching the existing
    SOC predictor. The first ``calibration_steps`` valid samples determine one
    latent representation per trajectory. Capacity and charge/discharge
    efficiencies derived from that representation are shared by every output
    step, while a causal recurrent branch models bounded dynamic corrections.

    The returned dictionary is intended to support multi-point supervision and
    uncertainty-aware downstream feedback. Padded positions are set to zero and
    identified by ``valid_mask``.
    """

    def __init__(
        self,
        hidden_dim=64,
        latent_dim=32,
        conv_layers=2,
        gru_layers=2,
        dropout=0.05,
        calibration_steps=64,
        dt_seconds=10.0,
        voltage_mean=350.0,
        voltage_std=30.0,
        rate_mean=0.0,
        rate_std=100.0,
        cumulative_ah_scale=20.0,
        init_capacity_ah=150.0,
        capacity_log_range=0.7,
        efficiency_log_range=0.15,
        physics_gain_log_range=0.5,
        use_current_correction=False,
        current_gain_log_range=0.25,
        current_bias_max_amp=2.0,
        max_dynamic_residual=0.08,
        min_log_variance=-9.0,
        max_log_variance=-2.0,
        time_scale_steps=None
    ):
        super().__init__()
        if hidden_dim <= 0 or latent_dim <= 0:
            raise ValueError("hidden_dim and latent_dim must be positive")
        if calibration_steps <= 0:
            raise ValueError("calibration_steps must be positive")
        if dt_seconds <= 0 or cumulative_ah_scale <= 0 or init_capacity_ah <= 0:
            raise ValueError("physical scale parameters must be positive")
        if max_dynamic_residual <= 0:
            raise ValueError("max_dynamic_residual must be positive")
        if min_log_variance >= max_log_variance:
            raise ValueError("min_log_variance must be less than max_log_variance")

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.calibration_steps = calibration_steps
        self.dt_hours = dt_seconds / 3600.0
        self.cumulative_ah_scale = cumulative_ah_scale
        self.capacity_log_range = capacity_log_range
        self.efficiency_log_range = efficiency_log_range
        self.physics_gain_log_range = physics_gain_log_range
        self.use_current_correction = use_current_correction
        self.current_gain_log_range = current_gain_log_range
        self.current_bias_max_amp = current_bias_max_amp
        self.max_dynamic_residual = max_dynamic_residual
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance
        self.time_scale_steps = time_scale_steps

        self.register_buffer("voltage_mean", torch.tensor(float(voltage_mean)))
        self.register_buffer("voltage_std", torch.tensor(float(voltage_std)))
        self.register_buffer("rate_mean", torch.tensor(float(rate_mean)))
        self.register_buffer("rate_std", torch.tensor(float(rate_std)))

        input_dim = 9
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.conv = nn.Sequential(*[
            ResidualCausalConvBlock(
                hidden_dim,
                kernel_size=3,
                dilation=2 ** layer_index,
                dropout=dropout
            )
            for layer_index in range(conv_layers)
        ])
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )

        self.trajectory_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
        self.capacity_head = nn.Linear(latent_dim, 1)
        self.efficiency_head = nn.Linear(latent_dim, 2)
        self.current_correction_head = nn.Linear(latent_dim, 3)

        dynamic_dim = hidden_dim + latent_dim + 5
        self.physics_gain_head = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.dynamic_head = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.log_variance_head = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        init_inv_capacity = 1.0 / init_capacity_ah
        self.raw_inv_capacity = nn.Parameter(torch.tensor(
            math.log(math.expm1(init_inv_capacity)), dtype=torch.float32
        ))
        nn.init.zeros_(self.physics_gain_head[-1].weight)
        nn.init.zeros_(self.physics_gain_head[-1].bias)
        nn.init.zeros_(self.current_correction_head.weight)
        nn.init.zeros_(self.current_correction_head.bias)

    @staticmethod
    def _normalize_lengths(lengths, batch_size, seq_len, device):
        if lengths is None:
            return torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=device
            )
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=device).view(-1)
        if lengths.numel() != batch_size:
            raise ValueError("lengths must contain one value per batch item")
        return lengths.clamp(min=1, max=seq_len)

    @staticmethod
    def _gather_steps(values, indices):
        gather_index = indices.view(-1, 1, *([1] * (values.dim() - 2)))
        gather_index = gather_index.expand(-1, 1, *values.shape[2:])
        return values.gather(1, gather_index).squeeze(1)

    def forward(self, voltage_seq, rate_seq, soc0, lengths=None):
        squeeze_output = False
        if voltage_seq.dim() == 1:
            voltage_seq = voltage_seq.unsqueeze(0)
            rate_seq = rate_seq.unsqueeze(0)
            squeeze_output = True
        if voltage_seq.dim() != 2 or rate_seq.shape != voltage_seq.shape:
            raise ValueError("voltage_seq and rate_seq must have matching [batch, time] shapes")

        batch_size, seq_len = voltage_seq.shape
        lengths = self._normalize_lengths(
            lengths, batch_size, seq_len, voltage_seq.device
        )
        soc0 = torch.as_tensor(
            soc0, dtype=voltage_seq.dtype, device=voltage_seq.device
        ).view(-1)
        if soc0.numel() == 1 and batch_size > 1:
            soc0 = soc0.expand(batch_size)
        if soc0.numel() != batch_size:
            raise ValueError("soc0 must contain one value per batch item")

        time_index = torch.arange(seq_len, device=voltage_seq.device).view(1, -1)
        valid_mask = time_index < lengths.view(-1, 1)
        valid_float = valid_mask.to(voltage_seq.dtype)

        voltage_norm = (voltage_seq - self.voltage_mean) / self.voltage_std
        rate_norm = (rate_seq - self.rate_mean) / self.rate_std
        voltage_norm = voltage_norm * valid_float
        rate_norm = rate_norm * valid_float

        zero = torch.zeros(
            batch_size, 1, dtype=voltage_seq.dtype, device=voltage_seq.device
        )
        delta_voltage = torch.cat([
            zero, voltage_seq[:, 1:] - voltage_seq[:, :-1]
        ], dim=1) / self.voltage_std
        delta_rate = torch.cat([
            zero, rate_seq[:, 1:] - rate_seq[:, :-1]
        ], dim=1) / self.rate_std
        delta_voltage = delta_voltage * valid_float
        delta_rate = delta_rate * valid_float

        current_ah = rate_seq * self.dt_hours * valid_float
        charge_ah = torch.cumsum(torch.clamp(-current_ah, min=0.0), dim=1)
        discharge_ah = torch.cumsum(torch.clamp(current_ah, min=0.0), dim=1)
        net_ah = discharge_ah - charge_ah
        soc0_feature = soc0.view(-1, 1).expand(-1, seq_len)
        if self.time_scale_steps is None:
            relative_time = (
                time_index.to(voltage_seq.dtype)
                / (lengths - 1).clamp(min=1).to(voltage_seq.dtype).view(-1, 1)
            ).clamp(max=1.0)
        else:
            relative_time = (
                time_index.to(voltage_seq.dtype) / self.time_scale_steps
            ).clamp(max=1.0).expand(batch_size, -1)

        features = torch.stack([
            voltage_norm,
            rate_norm,
            delta_voltage,
            delta_rate,
            charge_ah / self.cumulative_ah_scale,
            discharge_ah / self.cumulative_ah_scale,
            net_ah / self.cumulative_ah_scale,
            soc0_feature,
            relative_time
        ], dim=-1)
        encoded = self.conv(self.input_proj(features))

        packed = pack_padded_sequence(
            encoded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        recurrent, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )

        calibration_lengths = lengths.clamp(max=self.calibration_steps)
        calibration_index = calibration_lengths - 1
        calibration_hidden = self._gather_steps(recurrent, calibration_index)
        calibration_features = torch.stack([
            self._gather_steps(voltage_norm, calibration_index),
            self._gather_steps(rate_norm, calibration_index),
            self._gather_steps(charge_ah, calibration_index) / self.cumulative_ah_scale,
            self._gather_steps(discharge_ah, calibration_index) / self.cumulative_ah_scale,
            soc0
        ], dim=-1)
        trajectory_latent = self.trajectory_encoder(torch.cat([
            calibration_hidden, calibration_features
        ], dim=-1))

        base_inv_capacity = F.softplus(self.raw_inv_capacity)
        capacity_scale = torch.exp(
            self.capacity_log_range
            * torch.tanh(self.capacity_head(trajectory_latent).squeeze(-1))
        )
        inv_capacity = base_inv_capacity * capacity_scale
        efficiencies = torch.exp(
            self.efficiency_log_range * torch.tanh(
                self.efficiency_head(trajectory_latent)
            )
        )
        charge_efficiency = efficiencies[:, 0]
        discharge_efficiency = efficiencies[:, 1]

        correction_parameters = self.current_correction_head(trajectory_latent)
        current_gain_logs = self.current_gain_log_range * torch.tanh(
            correction_parameters[:, :2]
        )
        current_gains = torch.exp(current_gain_logs)
        current_bias_amp = self.current_bias_max_amp * torch.tanh(
            correction_parameters[:, 2]
        )

        if self.use_current_correction:
            current_gain = torch.where(
                rate_seq < 0,
                current_gains[:, 0].view(-1, 1),
                current_gains[:, 1].view(-1, 1)
            )
            corrected_current = (
                current_gain * rate_seq + current_bias_amp.view(-1, 1)
            ) * valid_float
            corrected_current_ah = corrected_current * self.dt_hours
            charge_ah = torch.cumsum(
                torch.clamp(-corrected_current_ah, min=0.0), dim=1
            )
            discharge_ah = torch.cumsum(
                torch.clamp(corrected_current_ah, min=0.0), dim=1
            )
            charge_efficiency = torch.ones_like(charge_efficiency)
            discharge_efficiency = torch.ones_like(discharge_efficiency)

        physical_delta = inv_capacity.view(-1, 1) * (
            charge_efficiency.view(-1, 1) * charge_ah
            - discharge_efficiency.view(-1, 1) * discharge_ah
        )
        latent_sequence = trajectory_latent.unsqueeze(1).expand(-1, seq_len, -1)
        dynamic_features = torch.cat([
            recurrent,
            latent_sequence,
            voltage_norm.unsqueeze(-1),
            rate_norm.unsqueeze(-1),
            (charge_ah / self.cumulative_ah_scale).unsqueeze(-1),
            (discharge_ah / self.cumulative_ah_scale).unsqueeze(-1),
            relative_time.unsqueeze(-1)
        ], dim=-1)
        if self.use_current_correction:
            physics_log_gain = torch.zeros_like(physical_delta)
            physics_gain = torch.ones_like(physical_delta)
            physical_contribution = physical_delta
        else:
            physics_log_gain = self.physics_gain_log_range * torch.tanh(
                self.physics_gain_head(dynamic_features).squeeze(-1)
            )
            physics_gain = torch.exp(physics_log_gain)
            physical_contribution = physics_gain * physical_delta
        dynamic_residual = self.max_dynamic_residual * torch.tanh(
            self.dynamic_head(dynamic_features).squeeze(-1)
        )
        soc = torch.clamp(
            soc0.view(-1, 1) + physical_contribution + dynamic_residual,
            min=0.0,
            max=1.0
        )

        raw_log_variance = self.log_variance_head(
            dynamic_features.detach()
        ).squeeze(-1)
        log_variance = self.min_log_variance + (
            self.max_log_variance - self.min_log_variance
        ) * torch.sigmoid(raw_log_variance)

        soc = soc * valid_float
        physical_delta = physical_delta * valid_float
        physical_contribution = physical_contribution * valid_float
        physics_gain = physics_gain * valid_float
        physics_log_gain = physics_log_gain * valid_float
        dynamic_residual = dynamic_residual * valid_float
        log_variance = log_variance * valid_float

        output = {
            "soc": soc,
            "physical_delta": physical_delta,
            "physical_contribution": physical_contribution,
            "physics_gain": physics_gain,
            "physics_log_gain": physics_log_gain,
            "charge_current_gain": current_gains[:, 0],
            "discharge_current_gain": current_gains[:, 1],
            "current_gain_logs": current_gain_logs,
            "current_bias_amp": current_bias_amp,
            "dynamic_residual": dynamic_residual,
            "log_variance": log_variance,
            "valid_mask": valid_mask,
            "trajectory_latent": trajectory_latent,
            "effective_capacity_ah": inv_capacity.reciprocal(),
            "charge_efficiency": charge_efficiency,
            "discharge_efficiency": discharge_efficiency,
            "calibration_lengths": calibration_lengths
        }
        if squeeze_output:
            output = {
                key: value.squeeze(0) if value.dim() > 0 else value
                for key, value in output.items()
            }
        return output


    def predict_endpoint(self, voltage_seq, rate_seq, soc0, lengths=None):
        output = self(voltage_seq, rate_seq, soc0, lengths)
        if output["soc"].dim() == 1:
            endpoint_index = output["valid_mask"].sum() - 1
            return output["soc"][endpoint_index]
        normalized_lengths = self._normalize_lengths(
            lengths,
            output["soc"].shape[0],
            output["soc"].shape[1],
            output["soc"].device
        )
        return self._gather_steps(output["soc"], normalized_lengths - 1)
