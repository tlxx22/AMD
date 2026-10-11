"""Reuse unchanged existing assertions with explicit generated CSV fixtures."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

from tests.urbanev_synthetic_fixture import write_synthetic_urbanev
from utils.dataloader_urbanev import UrbanEVRawData


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root=Path(__file__).resolve().parents[1]
    names=['tests.test_fold_scaler_no_leakage','tests.test_target_offset']
    names.extend('tests.test_m6_native_time_marks.NativeTimeMarkTests.'+n for n in (
        'test_historical_window_no_future_leak', 'test_native_MS_calls_and_TimeXer_channel_reorder',
        'test_M_full_output_original_column_order', 'test_other_model_calls_and_marks_rejection',
        'test_missing_wrong_mark_scope_shape_rejected'))
    with tempfile.TemporaryDirectory(dir=root,prefix='.uv-synthetic-regression-') as temp:
        fixture=Path(temp)/'generated-csv'
        write_synthetic_urbanev(fixture)
        original_load=UrbanEVRawData.load
        def synthetic_load(cls, requested_root):
            # Existing test paths are redirected only inside this test process.
            # Real production parser still validates the generated 4344x275 CSV.
            return original_load(fixture)
        with patch.object(UrbanEVRawData,'load',classmethod(synthetic_load)):
            suite=unittest.defaultTestLoader.loadTestsFromNames(names)
            result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
