import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionRestorer(nn.Module):
    def __init__(
        self,
        pre_len=12,
        sub_len=6,
        max_missing_len=7,
        hidden_size=64,
        num_layers=1,
        dropout=0.1,
        pre_mean=None,
        pre_std=None,
        sub_mean=None,
        sub_std=None,
        rate_mean=0.0,
        rate_std=1.0,
        residual_scale=2.0,
    ):
        super().__init__()

        self.pre_len = pre_len
        self.sub_len = sub_len
        self.max_missing_len = max_missing_len
        self.hidden_size = hidden_size

        self.register_buffer("pre_mean", self._make_tensor(pre_mean, [0.0, 0.0, 0.0]))
        self.register_buffer("pre_std", self._make_tensor(pre_std, [1.0, 1.0, 1.0]))
        self.register_buffer("sub_mean", self._make_tensor(sub_mean, [0.0, 0.0]))
        self.register_buffer("sub_std", self._make_tensor(sub_std, [1.0, 1.0]))
        self.register_buffer("rate_mean", torch.tensor(float(rate_mean)))
        self.register_buffer("rate_std", torch.tensor(float(rate_std)))

        self.pre_proj = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        self.sub_proj = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        self.pre_encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.sub_encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.len_emb = nn.Embedding(max_missing_len + 1, hidden_size)
        self.step_emb = nn.Embedding(max_missing_len, hidden_size)

        boundary_dim = 15

        self.context_proj = nn.Sequential(
            nn.Linear(hidden_size * 3 + boundary_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2 + 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def _make_tensor(value, default):
        if value is None:
            value = default
        return torch.tensor(value, dtype=torch.float32)

    def forward(self, pre_seq, sub_seq, missing_len, return_mask=False):
        b = pre_seq.size(0)
        device = pre_seq.device
        dtype = pre_seq.dtype

        if not torch.is_tensor(missing_len):
            missing_len = torch.full((b,), int(missing_len), device=device, dtype=torch.long)
        else:
            missing_len = missing_len.to(device=device, dtype=torch.long).view(-1)

        missing_len = missing_len.clamp(1, self.max_missing_len)

        pre_mean = self.pre_mean.to(device=device, dtype=dtype)
        pre_std = self.pre_std.to(device=device, dtype=dtype).clamp_min(1e-6)
        sub_mean = self.sub_mean.to(device=device, dtype=dtype)
        sub_std = self.sub_std.to(device=device, dtype=dtype).clamp_min(1e-6)
        rate_mean = self.rate_mean.to(device=device, dtype=dtype)
        rate_std = self.rate_std.to(device=device, dtype=dtype).clamp_min(1e-6)

        pre_x = (pre_seq - pre_mean) / pre_std
        sub_x = (sub_seq - sub_mean) / sub_std

        pre_h = self.pre_proj(pre_x)
        sub_h = self.sub_proj(sub_x)

        _, pre_state = self.pre_encoder(pre_h)
        _, sub_state = self.sub_encoder(sub_h)

        pre_state = pre_state[-1]
        sub_state = sub_state[-1]

        pre_last = pre_x[:, -1]
        pre_delta = pre_x[:, -1] - pre_x[:, -2]
        pre_mean_feat = pre_x.mean(dim=1)

        sub_first = sub_x[:, 0]
        sub_delta = sub_x[:, 1] - sub_x[:, 0]
        sub_mean_feat = sub_x.mean(dim=1)

        boundary = torch.cat(
            [
                pre_last,
                pre_delta,
                pre_mean_feat,
                sub_first,
                sub_delta,
                sub_mean_feat,
            ],
            dim=-1,
        )

        len_feature = self.len_emb(missing_len)

        context = torch.cat(
            [
                pre_state,
                sub_state,
                len_feature,
                boundary,
            ],
            dim=-1,
        )

        context = self.context_proj(context)

        steps = torch.arange(self.max_missing_len, device=device).view(1, -1).expand(b, -1)
        mask = steps < missing_len.view(-1, 1)

        rel_pos = (steps.to(dtype) + 1.0) / (missing_len.view(-1, 1).to(dtype) + 1.0)

        last_rate = pre_seq[:, -1, 0:1]
        first_sub_rate = sub_seq[:, 0, 0:1]
        base_rate = last_rate + (first_sub_rate - last_rate) * rel_pos

        base_rate_norm = (base_rate - rate_mean) / rate_std

        step_feature = self.step_emb(steps)
        context_feature = context.unsqueeze(1).expand(-1, self.max_missing_len, -1)

        decoder_in = torch.cat(
            [
                context_feature,
                step_feature,
                rel_pos.unsqueeze(-1),
                base_rate_norm.unsqueeze(-1),
            ],
            dim=-1,
        )

        residual_norm = torch.tanh(self.decoder(decoder_in).squeeze(-1))
        residual = residual_norm * rate_std * self.residual_scale.abs()

        pred_rate = base_rate + residual
        pred_rate = pred_rate * mask.to(dtype)

        if return_mask:
            return pred_rate, mask

        return pred_rate