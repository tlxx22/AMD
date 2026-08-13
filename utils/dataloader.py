from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def _positive_int(name, value):
    """Validate integer loader parameters without accepting bool as an int."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


def _dataset_kind(dataset_stem):
    stem = dataset_stem.lower()
    if stem.startswith("pems"):
        return "pems"
    if stem.startswith("solar"):
        return "solar"
    if stem.startswith("ettm"):
        return "ettm"
    if stem.startswith("etth"):
        return "etth"
    return "generic"


def _compute_split_endpoints(dataset_stem, row_count):
    """Return the public-code train/validation/test split endpoints."""
    row_count = _positive_int("row_count", row_count)
    kind = _dataset_kind(dataset_stem)

    if kind == "ettm":
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = val_end + 4 * 30 * 24 * 4
    elif kind == "etth":
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = val_end + 4 * 30 * 24
    elif kind == "pems":
        train_end = int(row_count * 0.6)
        val_end = row_count - int(row_count * 0.2)
        test_end = row_count
    else:
        train_end = int(row_count * 0.7)
        val_end = row_count - int(row_count * 0.2)
        test_end = row_count

    if not 0 < train_end < val_end < test_end <= row_count:
        raise ValueError(
            "dataset is too short for the public split: "
            f"kind={kind}, rows={row_count}, endpoints="
            f"({train_end}, {val_end}, {test_end})"
        )
    return train_end, val_end, test_end


def _serializable_columns(columns):
    values = []
    for value in columns:
        if isinstance(value, np.generic):
            value = value.item()
        values.append(value)
    return values


def _resolve_target_column(columns, target):
    """Resolve a CLI string target against string or integer column labels."""
    if target in columns:
        return target

    # Headerless Solar and PEMS frames use integer labels, while argparse
    # supplies targets as strings.  Match only the exact string form and only
    # when that mapping identifies one column unambiguously.
    matches = [column for column in columns if str(column) == target]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"target {target!r} ambiguously matches columns {matches!r}"
        )
    return None


class CustomDataLoader:
    """Generate validated data loaders from the public AMD data format."""

    def __init__(
            self,
            data,
            batch_size,
            seq_len,
            pred_len,
            feature_type,
            target='OT',
            train_generator=None,
            dataset_id=None,
    ):
        self.data = Path(data).expanduser()
        if not self.data.is_file():
            raise FileNotFoundError(f"dataset file does not exist: {self.data}")

        self.batch_size = _positive_int("batch_size", batch_size)
        self.seq_len = _positive_int("seq_len", seq_len)
        self.pred_len = _positive_int("pred_len", pred_len)

        if feature_type not in {'S', 'M', 'MS'}:
            raise ValueError(
                "feature_type must be one of 'S', 'M', or 'MS', "
                f"got {feature_type!r}"
            )
        if not isinstance(target, str) or not target:
            raise ValueError("target must be a non-empty string")
        if dataset_id is None:
            dataset_id = self.data.stem
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string or None")
        self.feature_type = feature_type
        self.target = target
        self.dataset_id = dataset_id.strip()
        self.resolved_target = None
        self.target_slice = slice(0, None)
        self._train_generator = None
        self.train_generator = train_generator

        self._read_data()

    @staticmethod
    def _validate_generator(generator):
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError(
                "train_generator must be a torch.Generator or None, "
                f"got {type(generator).__name__}"
            )
        if generator is not None and generator.device.type != 'cpu':
            raise ValueError("train_generator must be a CPU torch.Generator")
        return generator

    @property
    def train_generator(self):
        return self._train_generator

    @train_generator.setter
    def train_generator(self, generator):
        self._train_generator = self._validate_generator(generator)

    def get_train_generator_state(self):
        """Return a checkpoint-safe copy of the train generator state."""
        if self._train_generator is None:
            return None
        return self._train_generator.get_state().clone()

    def set_train_generator_state(self, state):
        """Restore, or lazily create, the explicit train generator."""
        if not isinstance(state, torch.Tensor):
            raise TypeError("generator state must be a torch.Tensor")
        if state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("generator state must be a one-dimensional uint8 tensor")
        if self._train_generator is None:
            self._train_generator = torch.Generator()
        self._train_generator.set_state(state.detach().cpu())

    @property
    def preprocessing_metadata(self):
        """Return JSON-serializable preprocessing state without mutable aliases."""
        return deepcopy(self._preprocessing_metadata)

    def get_preprocessing_metadata(self):
        """Backward-friendly method form of :attr:`preprocessing_metadata`."""
        return self.preprocessing_metadata

    def metadata(self):
        """Return preprocessing metadata in the form consumed by the runner."""
        return self.preprocessing_metadata

    def _read_raw_dataframe(self):
        kind = _dataset_kind(self.dataset_id)
        if kind == 'pems':
            with np.load(self.data) as archive:
                if 'data' not in archive:
                    raise ValueError(f"PEMS archive has no 'data' array: {self.data}")
                data_raw = archive['data']
                if data_raw.ndim != 3 or data_raw.shape[2] < 1:
                    raise ValueError(
                        "PEMS 'data' array must have shape [time, feature, channel], "
                        f"got {data_raw.shape}"
                    )
                df = pd.DataFrame(data_raw[:, :, 0])
        elif kind == 'solar':
            # Solar-Energy is headerless; the first observation is real data.
            df = pd.read_csv(self.data, header=None)
        else:
            df_raw = pd.read_csv(self.data)
            if 'date' not in df_raw.columns:
                raise ValueError(
                    f"non-Solar CSV must contain a 'date' column: {self.data}"
                )
            df = df_raw.set_index('date')

        if df.empty:
            raise ValueError(f"dataset contains no observations: {self.data}")
        if df.shape[1] == 0:
            raise ValueError(f"dataset contains no feature columns: {self.data}")

        try:
            df = df.apply(pd.to_numeric, errors='raise')
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"dataset feature values must all be numeric: {self.data}"
            ) from exc

        values = df.to_numpy(copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"dataset contains NaN or infinite feature values: {self.data}")
        return df, kind

    def _read_data(self):
        df, kind = self._read_raw_dataframe()
        raw_row_count = len(df)

        if self.feature_type in {'S', 'MS'}:
            self.resolved_target = _resolve_target_column(df.columns, self.target)
            if self.resolved_target is None:
                raise ValueError(
                    f"target column {self.target!r} is missing from dataset {self.data}"
                )
        if self.feature_type == 'S':
            df = df[[self.resolved_target]]
            self.target_slice = slice(0, 1)
        elif self.feature_type == 'MS':
            target_idx = int(df.columns.get_loc(self.resolved_target))
            self.target_slice = slice(target_idx, target_idx + 1)

        train_end, val_end, test_end = _compute_split_endpoints(
            self.dataset_id, raw_row_count
        )
        val_start_with_context = train_end - self.seq_len
        test_start_with_context = val_end - self.seq_len

        window_counts = {
            'train': train_end - self.seq_len - self.pred_len + 1,
            'val': (val_end - train_end) - self.pred_len + 1,
            'test': (test_end - val_end) - self.pred_len + 1,
        }
        invalid_splits = {
            name: count for name, count in window_counts.items() if count <= 0
        }
        if val_start_with_context < 0 or test_start_with_context < 0 or invalid_splits:
            raise ValueError(
                "seq_len/pred_len leave an empty split: "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}, "
                f"window_counts={window_counts}"
            )
        if window_counts['train'] < self.batch_size:
            raise ValueError(
                "training split has fewer windows than batch_size while "
                "drop_last=True: "
                f"train_windows={window_counts['train']}, "
                f"batch_size={self.batch_size}"
            )

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start_with_context:val_end]
        test_df = df.iloc[test_start_with_context:test_end]

        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(frame):
            data = self.scaler.transform(frame.values)
            return pd.DataFrame(data, index=frame.index, columns=frame.columns)

        self.train_df = scale_df(train_df)
        self.val_df = scale_df(val_df)
        self.test_df = scale_df(test_df)
        self.n_feature = int(self.train_df.shape[-1])
        self.split_endpoints = {
            'train_end': int(train_end),
            'val_end': int(val_end),
            'test_end': int(test_end),
        }
        self.window_counts = {key: int(value) for key, value in window_counts.items()}

        target_slice_metadata = {
            'start': self.target_slice.start,
            'stop': self.target_slice.stop,
            'step': self.target_slice.step,
        }
        if self.target_slice.stop is None:
            target_indices = list(range(self.n_feature))
        else:
            target_indices = list(range(
                self.target_slice.start,
                self.target_slice.stop,
                self.target_slice.step or 1,
            ))

        self._preprocessing_metadata = {
            'data_path': str(self.data.resolve()),
            'dataset_stem': self.data.stem,
            'dataset_id': self.dataset_id,
            'dataset_kind': kind,
            'raw_rows': int(raw_row_count),
            'used_rows': int(test_end),
            'feature_type': self.feature_type,
            'target': self.target,
            'resolved_target': (
                _serializable_columns([self.resolved_target])[0]
                if self.resolved_target is not None else None
            ),
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'batch_size': self.batch_size,
            'columns': _serializable_columns(self.train_df.columns),
            'target_slice': target_slice_metadata,
            'target_indices': target_indices,
            'scaler': {
                'mean': [float(value) for value in self.scaler.mean_],
                'scale': [float(value) for value in self.scaler.scale_],
            },
            'split_endpoints': deepcopy(self.split_endpoints),
            'split_context_starts': {
                'train': 0,
                'val': int(val_start_with_context),
                'test': int(test_start_with_context),
            },
            'window_counts': deepcopy(self.window_counts),
        }

        # Report the number of real (input, target) windows, not raw split rows.
        print("train : ", self.window_counts['train'])
        print("valid : ", self.window_counts['val'])
        print("test  : ", self.window_counts['test'])

    def _make_dataset(self, data, shuffle, drop_last, generator=None, split_name=None):
        array = np.asarray(data, dtype=np.float32)
        data_x = torch.from_numpy(array)
        data_y = torch.from_numpy(array[:, self.target_slice])
        dataset = CustomDataset(data_x, data_y, self.seq_len, self.pred_len)

        if split_name is not None:
            expected = self.window_counts[split_name]
            if len(dataset) != expected:
                raise RuntimeError(
                    f"{split_name} window count mismatch: "
                    f"expected {expected}, got {len(dataset)}"
                )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=generator,
        )

    def inverse_transform(self, data):
        """Undo scaling for full features or an S/MS target-only array.

        Leading dimensions are preserved; the last dimension identifies which
        scaler parameters apply.  M tasks require the full feature width.
        """
        array = np.asarray(data)
        if array.ndim == 0:
            raise ValueError("inverse_transform data must have a feature dimension")

        width = int(array.shape[-1])
        full_width = self.n_feature
        target_indices = self._preprocessing_metadata['target_indices']
        target_width = len(target_indices)

        if width == full_width:
            indices = np.arange(full_width, dtype=np.int64)
        elif self.feature_type in {'S', 'MS'} and width == target_width:
            indices = np.asarray(target_indices, dtype=np.int64)
        else:
            expected = [full_width]
            if self.feature_type in {'S', 'MS'} and target_width != full_width:
                expected.append(target_width)
            raise ValueError(
                "inverse_transform last dimension has unsupported width: "
                f"got {width}, expected one of {expected} for "
                f"feature_type={self.feature_type!r}"
            )

        return (
            array * self.scaler.scale_[indices]
            + self.scaler.mean_[indices]
        )

    def get_train(self, generator=None):
        if generator is not None:
            self.train_generator = generator
        return self._make_dataset(
            self.train_df,
            shuffle=True,
            drop_last=True,
            generator=self._train_generator,
            split_name='train',
        )

    def get_val(self, shuffle=False):
        if shuffle is not False:
            raise ValueError("validation shuffle must remain False")
        return self._make_dataset(
            self.val_df,
            shuffle=False,
            drop_last=False,
            split_name='val',
        )

    def get_test(self):
        return self._make_dataset(
            self.test_df,
            shuffle=False,
            drop_last=False,
            split_name='test',
        )


class CustomDataset(Dataset):
    def __init__(self, data_x, data_y, seq_len, pred_len):
        if not isinstance(data_x, torch.Tensor) or not isinstance(data_y, torch.Tensor):
            raise TypeError("data_x and data_y must be torch.Tensor instances")
        if data_x.ndim < 1 or data_y.ndim < 1:
            raise ValueError("data_x and data_y must have a time dimension")
        if data_x.shape[0] != data_y.shape[0]:
            raise ValueError(
                "data_x and data_y must contain the same number of time points"
            )
        self.data_x = data_x
        self.data_y = data_y
        self.seq_len = _positive_int("seq_len", seq_len)
        self.pred_len = _positive_int("pred_len", pred_len)
        self._window_count = max(
            0, int(self.data_x.shape[0]) - self.seq_len - self.pred_len + 1
        )

    def __len__(self):
        return self._window_count

    def __getitem__(self, idx):
        if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
            raise TypeError(f"dataset index must be an integer, got {idx!r}")
        idx = int(idx)
        if idx < 0 or idx >= self._window_count:
            raise IndexError(
                f"dataset index {idx} is out of range for {self._window_count} windows"
            )
        return (
            self.data_x[idx: idx + self.seq_len],
            self.data_y[
                idx + self.seq_len: idx + self.seq_len + self.pred_len
            ],
        )
