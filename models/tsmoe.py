import math

import torch
import torch.nn as nn


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_input_shape(input_shape, module_name):
    if not isinstance(input_shape, (tuple, list, torch.Size)) or len(input_shape) != 2:
        raise ValueError(
            f"{module_name} input_shape must be (seq_len, feature_num), "
            f"got {input_shape!r}"
        )
    seq_len = _positive_int(input_shape[0], f"{module_name} seq_len")
    feature_num = _positive_int(input_shape[1], f"{module_name} feature_num")
    return seq_len, feature_num


class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_epsilon=1e-5):
        super(TopKGating, self).__init__()
        input_dim = _positive_int(input_dim, "TopKGating input_dim")
        num_experts = _positive_int(num_experts, "TopKGating num_experts")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= num_experts:
            raise ValueError(
                f"TopKGating top_k must be in [1, {num_experts}], got {top_k!r}"
            )
        if (
            isinstance(noise_epsilon, bool)
            or not isinstance(noise_epsilon, (int, float))
            or not math.isfinite(noise_epsilon)
            or noise_epsilon < 0
        ):
            raise ValueError(
                "TopKGating noise_epsilon must be finite and non-negative, "
                f"got {noise_epsilon!r}"
            )
        self.input_dim = input_dim
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        self.num_experts = num_experts
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

    def decompostion_tp(self, x, alpha=10):
        # x [batch_size, seq_len]
        if not torch.is_tensor(x):
            raise TypeError("TopKGating logits must be a torch.Tensor")
        if x.ndim != 2 or x.shape[1] != self.num_experts:
            raise ValueError(
                "TopKGating decomposition expects "
                f"[batch, {self.num_experts}], got {tuple(x.shape)}"
            )
        output = torch.zeros_like(x)
        # [batch_size]
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)
        # [batch_size, num_expert]
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)
        mask = x < kth_largest_mat
        x = self.softmax(x)
        output[mask] = alpha * torch.log(x[mask] + 1)
        output[~mask] = alpha * (torch.exp(x[~mask]) - 1)
        # Ablation Spare MoE
        # output[mask] = 0
        # [batch_size, seq_len]
        return output

    def forward(self, x):
        # [batch_size, seq_len]

        if not torch.is_tensor(x):
            raise TypeError("TopKGating input must be a torch.Tensor")
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"TopKGating expects [batch, {self.input_dim}], got {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError("TopKGating requires a non-empty batch")

        x = self.gate(x)
        clean_logits = x
        # [batch_size, num_experts]

        if self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.decompostion_tp(logits)
        gates = self.softmax(logits)

        # random order
        # indices = torch.randperm(gates.size(0))
        # shuffled_gates = gates[indices]

        # average
        # value = 1.0 / x.shape[1]
        # gates = torch.full(x.shape, value, device=x.device)

        return gates


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.2):
        super(Expert, self).__init__()
        input_dim = _positive_int(input_dim, "Expert input_dim")
        output_dim = _positive_int(output_dim, "Expert output_dim")
        hidden_dim = _positive_int(hidden_dim, "Expert hidden_dim")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError(
                f"Expert dropout must satisfy 0 <= dropout < 1, got {dropout!r}"
            )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class AMS(nn.Module):
    def __init__(self, input_shape, pred_len, ff_dim=2048, dropout=0.2, loss_coef=1.0, num_experts=4, top_k=2):
        super(AMS, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.seq_len, self.feature_num = _validate_input_shape(input_shape, "AMS")
        pred_len = _positive_int(pred_len, "AMS pred_len")
        ff_dim = _positive_int(ff_dim, "AMS ff_dim")
        num_experts = _positive_int(num_experts, "AMS num_experts")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= num_experts:
            raise ValueError(f"AMS top_k must be in [1, {num_experts}], got {top_k!r}")
        if (
            isinstance(loss_coef, bool)
            or not isinstance(loss_coef, (int, float))
            or not math.isfinite(loss_coef)
            or loss_coef < 0
        ):
            raise ValueError(
                f"AMS loss_coef must be finite and non-negative, got {loss_coef!r}"
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.pred_len = pred_len

        self.gating = TopKGating(self.seq_len, num_experts, top_k)

        self.experts = nn.ModuleList(
            [Expert(self.seq_len, pred_len, hidden_dim=ff_dim, dropout=dropout) for _ in range(num_experts)])
        self.loss_coef = loss_coef

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, time_embedding):
        # [batch_size, feature_num, seq_len]
        if not torch.is_tensor(x) or not torch.is_tensor(time_embedding):
            raise TypeError("AMS x and time_embedding must both be torch.Tensor instances")
        if x.ndim != 3 or time_embedding.ndim != 3:
            raise ValueError(
                "AMS expects x and time_embedding shaped [batch, feature, sequence], "
                f"got {tuple(x.shape)} and {tuple(time_embedding.shape)}"
            )
        if x.shape != time_embedding.shape:
            raise ValueError(
                "AMS x and time_embedding must have identical shapes, "
                f"got {tuple(x.shape)} and {tuple(time_embedding.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError("AMS requires a non-empty batch")
        if x.shape[1] != self.feature_num or x.shape[2] != self.seq_len:
            raise ValueError(
                f"AMS expects [batch, {self.feature_num}, {self.seq_len}], "
                f"got {tuple(x.shape)}"
            )
        if x.device != time_embedding.device:
            raise ValueError(
                "AMS x and time_embedding must be on the same device, "
                f"got {x.device} and {time_embedding.device}"
            )
        if x.dtype != time_embedding.dtype:
            raise ValueError(
                "AMS x and time_embedding must have the same dtype, "
                f"got {x.dtype} and {time_embedding.dtype}"
            )
        batch_size = x.shape[0]
        feature_num = x.shape[1]
        # [feature_num, batch_size, seq_len]
        x = torch.transpose(x, 0, 1)
        time_embedding = torch.transpose(time_embedding, 0, 1)

        output = x.new_zeros((feature_num, batch_size, self.pred_len))
        loss = 0

        for i in range(feature_num):
            input = x[i]
            time_info = time_embedding[i]
            # x[i]  [batch_size, seq_len]
            gates = self.gating(time_info)

            # expert_outputs [batch_size, num_experts, pred_len]
            expert_outputs = x.new_zeros((self.num_experts, batch_size, self.pred_len))

            for j in range(self.num_experts):
                expert_outputs[j, :, :] = self.experts[j](input)
            expert_outputs = torch.transpose(expert_outputs, 0, 1)
            # gates [batch_size, num_experts, pred_len]
            gates = gates.unsqueeze(-1).expand(-1, -1, self.pred_len)
            # batch_output [batch_size, pred_len]
            batch_output = (gates * expert_outputs).sum(1)
            output[i, :, :] = batch_output

            importance = gates.sum(0)
            loss += self.loss_coef * self.cv_squared(importance)

        # [feature_num, batch_size, seq_len]
        output = torch.transpose(output, 0, 1)
        # [batch_size, feature_num, seq_len]

        return output, loss
