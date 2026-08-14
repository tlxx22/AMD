import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def _validate_input_shape(input_shape, module_name):
    if not isinstance(input_shape, (tuple, list, torch.Size)) or len(input_shape) != 2:
        raise ValueError(
            f"{module_name} input_shape must be (seq_len, feature_num), "
            f"got {input_shape!r}"
        )
    seq_len, feature_num = input_shape
    for name, value in (("seq_len", seq_len), ("feature_num", feature_num)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{module_name} {name} must be a positive integer, got {value!r}"
            )
    return seq_len, feature_num


def _validate_bcl_input(x, seq_len, feature_num, module_name):
    if not torch.is_tensor(x):
        raise TypeError(f"{module_name} input must be a torch.Tensor")
    if x.ndim != 3:
        raise ValueError(
            f"{module_name} expects [batch, feature, sequence], got shape {tuple(x.shape)}"
        )
    if x.shape[0] <= 0:
        raise ValueError(f"{module_name} requires a non-empty batch")
    if x.shape[1] != feature_num or x.shape[2] != seq_len:
        raise ValueError(
            f"{module_name} expects [batch, {feature_num}, {seq_len}], "
            f"got {tuple(x.shape)}"
        )


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        if isinstance(num_features, bool) or not isinstance(num_features, int) or num_features <= 0:
            raise ValueError(f"RevIN num_features must be a positive integer, got {num_features!r}")
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, target_slice=None):
        if not torch.is_tensor(x):
            raise TypeError("RevIN input must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(
                f"RevIN expects [batch, sequence, feature], got shape {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError("RevIN requires a non-empty batch")
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"RevIN expects {self.num_features} features, got {x.shape[-1]}"
            )
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            if not hasattr(self, 'mean') or not hasattr(self, 'stdev'):
                raise RuntimeError("RevIN denorm requires a preceding norm call")
            if x.shape[0] != self.mean.shape[0]:
                raise ValueError(
                    "RevIN denorm batch size must match the preceding norm call: "
                    f"expected {self.mean.shape[0]}, got {x.shape[0]}"
                )
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps * self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x


class MDM(nn.Module):
    def __init__(self, input_shape, k=3, c=2, layernorm=True):
        super(MDM, self).__init__()
        self.seq_len, self.feature_num = _validate_input_shape(input_shape, "MDM")
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError(f"MDM k must be a non-negative integer, got {k!r}")
        if isinstance(c, bool) or not isinstance(c, int) or c < 1:
            raise ValueError(f"MDM c must be a positive integer, got {c!r}")
        if k > 0 and c ** k > self.seq_len:
            raise ValueError(
                f"MDM largest pooling scale c**k ({c ** k}) cannot exceed "
                f"seq_len ({self.seq_len})"
            )
        if not isinstance(layernorm, bool):
            raise TypeError(f"MDM layernorm must be bool, got {type(layernorm).__name__}")
        self.k = k
        if self.k > 0:
            self.k_list = [c ** i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.seq_len // k, self.seq_len // k),
                                  nn.GELU(),
                                  nn.Linear(self.seq_len // k, self.seq_len * c // k),
                                  )
                    for k in self.k_list
                ]
            )
        self.layernorm = layernorm
        if self.layernorm:
            # Variant-defined MDM-entry LayerNorm normalizes input X over its
            # final look-back dimension in [batch, channel, sequence].
            self.norm = nn.LayerNorm(self.seq_len)

    def forward(self, x):
        _validate_bcl_input(x, self.seq_len, self.feature_num, "MDM")
        if self.layernorm:
            x = self.norm(x)
        if self.k == 0:
            return x
        # x [batch_size, feature_num, seq_len]
        sample_x = []
        for i, k in enumerate(self.k_list):
            sample_x.append(self.avg_pools[i](x))
        sample_x.append(x)
        n = len(sample_x)
        for i in range(n - 1):
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)
        # [batch_size, feature_num, seq_len]
        return sample_x[n - 1]


class DDI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=12, alpha=0.0, layernorm=True):
        super(DDI, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        seq_len, feature_num = _validate_input_shape(input_shape, "DDI")
        if isinstance(patch, bool) or not isinstance(patch, int) or patch <= 0:
            raise ValueError(f"DDI patch must be a positive integer, got {patch!r}")
        if patch > seq_len:
            raise ValueError(f"DDI patch ({patch}) cannot exceed seq_len ({seq_len})")
        if seq_len % patch != 0:
            raise ValueError(
                f"DDI requires seq_len divisible by patch, got seq_len={seq_len}, "
                f"patch={patch}"
            )
        if not isinstance(layernorm, bool):
            raise TypeError(f"DDI layernorm must be bool, got {type(layernorm).__name__}")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or alpha < 0
        ):
            raise ValueError(f"DDI alpha must be finite and non-negative, got {alpha!r}")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError(f"DDI dropout must satisfy 0 <= dropout < 1, got {dropout!r}")
        self.input_shape = (seq_len, feature_num)
        self.seq_len = seq_len
        self.feature_num = feature_num
        if alpha > 0.0:
            self.ff_dim = max(
                32, 2 ** math.ceil(math.log2(self.input_shape[-1]))
            )
            self.fc_block = nn.Sequential(
                nn.Linear(self.input_shape[-1], self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.input_shape[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.n_history = 1
        self.alpha = alpha
        self.patch = patch

        self.layernorm = layernorm
        if self.layernorm:
            # Keep the public flag's scope while using actual LayerNorm over
            # the final sequence dimension. Internal norm1/norm2 below retain
            # their released BatchNorm1d semantics.
            self.norm = nn.LayerNorm(self.seq_len)
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])

        self.agg = nn.Linear(self.n_history * self.patch, self.patch)
        self.dropout_t = nn.Dropout(dropout)

    def forward(self, x):
        # [batch_size, feature_num, seq_len]
        _validate_bcl_input(x, self.seq_len, self.feature_num, "DDI")
        uses_batch_norm = self.seq_len > self.patch
        if self.training and uses_batch_norm and x.shape[0] < 2:
            raise ValueError(
                "DDI uses BatchNorm1d and requires training batch size >= 2"
            )
        if self.layernorm:
            x = self.norm(x)

        output = torch.zeros_like(x)
        output[:, :, :self.n_history * self.patch] = x[:, :, :self.n_history * self.patch].clone()
        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            # input [batch_size, feature_num, self.n_history * patch]
            input = output[:, :, i - self.n_history * self.patch: i]
            # input [batch_size, feature_num, self.n_history * patch]
            input = self.norm1(torch.flatten(input, 1, -1)).reshape(input.shape)
            # aggregation
            # [batch_size, feature_num, patch]
            input = F.gelu(self.agg(input))  # self.n_history * patch -> patch
            input = self.dropout_t(input)
            # input [batch_size, feature_num, patch]
            # input = torch.squeeze(input, dim=-1)
            tmp = input + x[:, :, i: i + self.patch]

            res = tmp

            # [batch_size, feature_num, patch]
            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)
                # [batch_size, patch, feature_num]
                tmp = self.fc_block(tmp)
                tmp = torch.transpose(tmp, 1, 2)
            output[:, :, i: i + self.patch] = res + self.alpha * tmp

        # [batch_size, feature_num, seq_len]
        return output

