import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataloader import (  # noqa: E402
    CustomDataLoader,
    CustomDataset,
    _compute_split_endpoints,
)


class DataLoaderTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, name, rows, columns=None):
        path = self.root / name
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            if columns is not None:
                writer.writerow(columns)
            writer.writerows(rows)
        return path

    @staticmethod
    def generic_rows(count, include_target=True):
        rows = []
        for index in range(count):
            row = [f'2024-01-01 {index:02d}:00:00', float(index)]
            if include_target:
                row.append(float(index * 2))
            rows.append(row)
        return rows

    def make_generic_loader(
            self,
            *,
            row_count=30,
            batch_size=4,
            seq_len=3,
            pred_len=2,
            feature_type='M',
            target='OT',
            generator=None,
    ):
        path = self.write_csv(
            'tiny.csv',
            self.generic_rows(row_count),
            ['date', 'a', 'OT'],
        )
        output = io.StringIO()
        with redirect_stdout(output):
            loader = CustomDataLoader(
                path,
                batch_size,
                seq_len,
                pred_len,
                feature_type,
                target,
                train_generator=generator,
            )
        return loader, output.getvalue()

    def test_custom_dataset_length_and_last_window(self):
        data_x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        data_y = data_x + 100
        dataset = CustomDataset(data_x, data_y, seq_len=3, pred_len=2)

        self.assertEqual(len(dataset), 6)
        x, y = dataset[5]
        self.assertTrue(torch.equal(x, data_x[5:8]))
        self.assertTrue(torch.equal(y, data_y[8:10]))
        with self.assertRaises(IndexError):
            _ = dataset[6]
        with self.assertRaises(IndexError):
            _ = dataset[-1]

    def test_public_split_endpoints_are_unchanged(self):
        self.assertEqual(
            _compute_split_endpoints('ETTh1', 17420),
            (8640, 11520, 14400),
        )
        self.assertEqual(
            _compute_split_endpoints('ETTm1', 69680),
            (34560, 46080, 57600),
        )
        self.assertEqual(_compute_split_endpoints('PEMS03', 100), (60, 80, 100))
        self.assertEqual(_compute_split_endpoints('weather', 100), (70, 80, 100))

    def test_real_window_counts_and_loader_policies(self):
        loader, printed = self.make_generic_loader()

        self.assertEqual(loader.window_counts, {'train': 17, 'val': 2, 'test': 5})
        self.assertIn('train :  17', printed)
        self.assertIn('valid :  2', printed)
        self.assertIn('test  :  5', printed)

        train = loader.get_train()
        val = loader.get_val()
        test = loader.get_test()

        self.assertIsInstance(train.dataset, CustomDataset)
        self.assertIsInstance(train.sampler, RandomSampler)
        self.assertTrue(train.drop_last)
        self.assertEqual(len(train.dataset), 17)
        self.assertEqual(len(train), 4)

        self.assertIsInstance(val.sampler, SequentialSampler)
        self.assertFalse(val.drop_last)
        self.assertEqual(len(val.dataset), 2)
        self.assertEqual(len(val), 1)

        self.assertIsInstance(test.sampler, SequentialSampler)
        self.assertFalse(test.drop_last)
        self.assertEqual(len(test.dataset), 5)
        self.assertEqual(len(test), 2)
        with self.assertRaises(ValueError):
            loader.get_val(shuffle=True)

    def test_solar_is_read_without_dropping_first_observation(self):
        rows = [
            [float(index), float(index + 100), float(index + 200)]
            for index in range(50)
        ]
        path = self.write_csv('solar_tiny.txt', rows, columns=None)
        with redirect_stdout(io.StringIO()):
            loader = CustomDataLoader(
                path,
                batch_size=4,
                seq_len=4,
                pred_len=2,
                feature_type='M',
            )

        metadata = loader.preprocessing_metadata
        self.assertEqual(metadata['dataset_kind'], 'solar')
        self.assertEqual(metadata['raw_rows'], 50)
        self.assertEqual(metadata['columns'], [0, 1, 2])
        self.assertEqual(metadata['window_counts'], {
            'train': 30,
            'val': 4,
            'test': 9,
        })
        # A correct 35-row train split includes values 0 through 34. If row zero
        # were parsed as a header, the shifted split would instead average 17.5.
        self.assertAlmostEqual(metadata['scaler']['mean'][0], 17.0)

    def test_dataset_id_controls_kind_when_file_is_renamed(self):
        rows = [
            [float(index), float(index + 100), float(index + 200)]
            for index in range(50)
        ]
        renamed_path = self.write_csv('renamed_data.txt', rows, columns=None)

        with self.assertRaisesRegex(ValueError, "'date' column"):
            CustomDataLoader(
                renamed_path, 4, 4, 2, 'M'
            )

        with redirect_stdout(io.StringIO()):
            loader = CustomDataLoader(
                renamed_path,
                batch_size=4,
                seq_len=4,
                pred_len=2,
                feature_type='M',
                dataset_id='solar_AL',
            )
        metadata = loader.metadata()
        self.assertEqual(metadata['dataset_stem'], 'renamed_data')
        self.assertEqual(metadata['dataset_id'], 'solar_AL')
        self.assertEqual(metadata['dataset_kind'], 'solar')
        self.assertEqual(metadata['raw_rows'], 50)
        self.assertEqual(metadata['window_counts'], {
            'train': 30,
            'val': 4,
            'test': 9,
        })

    def test_string_target_uniquely_resolves_solar_and_pems_integer_columns(self):
        solar_rows = [
            [float(index), float(index + 100), float(index + 200)]
            for index in range(50)
        ]
        solar_path = self.write_csv('solar_targets.txt', solar_rows, columns=None)

        with redirect_stdout(io.StringIO()):
            solar_s = CustomDataLoader(
                solar_path, 4, 4, 2, 'S', target='0'
            )
            solar_ms = CustomDataLoader(
                solar_path, 4, 4, 2, 'MS', target='0'
            )
        self.assertEqual(solar_s.metadata()['columns'], [0])
        self.assertEqual(solar_s.metadata()['resolved_target'], 0)
        self.assertEqual(solar_s.metadata()['target_indices'], [0])
        self.assertEqual(solar_ms.metadata()['columns'], [0, 1, 2])
        self.assertEqual(solar_ms.metadata()['resolved_target'], 0)
        self.assertEqual(solar_ms.metadata()['target_slice']['start'], 0)

        pems_path = self.root / 'renamed_pems.bin'
        pems_values = np.arange(50 * 3, dtype=np.float32).reshape(50, 3, 1)
        with pems_path.open('wb') as stream:
            np.savez(stream, data=pems_values)
        with redirect_stdout(io.StringIO()):
            pems_ms = CustomDataLoader(
                pems_path,
                batch_size=4,
                seq_len=4,
                pred_len=2,
                feature_type='MS',
                target='0',
                dataset_id='PEMS03',
            )
        pems_metadata = pems_ms.metadata()
        self.assertEqual(pems_metadata['dataset_kind'], 'pems')
        self.assertEqual(pems_metadata['resolved_target'], 0)
        self.assertEqual(pems_metadata['window_counts'], {
            'train': 25,
            'val': 9,
            'test': 9,
        })

    def test_preprocessing_metadata_is_json_serializable_and_defensive(self):
        loader, _ = self.make_generic_loader(feature_type='MS')
        metadata = loader.preprocessing_metadata

        json.dumps(metadata)
        self.assertEqual(loader.metadata(), metadata)
        self.assertEqual(metadata['columns'], ['a', 'OT'])
        self.assertEqual(metadata['target_slice'], {
            'start': 1,
            'stop': 2,
            'step': None,
        })
        self.assertEqual(metadata['target_indices'], [1])
        self.assertEqual(metadata['split_endpoints'], {
            'train_end': 21,
            'val_end': 24,
            'test_end': 30,
        })
        self.assertAlmostEqual(metadata['scaler']['mean'][0], 10.0)
        self.assertAlmostEqual(metadata['scaler']['mean'][1], 20.0)

        metadata['columns'].append('mutated')
        self.assertEqual(loader.preprocessing_metadata['columns'], ['a', 'OT'])

    def test_inverse_transform_accepts_full_or_s_ms_target_width(self):
        ms_loader, _ = self.make_generic_loader(feature_type='MS')

        full_standardized = np.zeros((2, 3, 2), dtype=np.float32)
        full_restored = ms_loader.inverse_transform(full_standardized)
        self.assertEqual(full_restored.shape, full_standardized.shape)
        np.testing.assert_allclose(
            full_restored,
            np.broadcast_to(np.array([10.0, 20.0]), (2, 3, 2)),
        )

        target_standardized = np.zeros((2, 3, 1), dtype=np.float32)
        target_restored = ms_loader.inverse_transform(target_standardized)
        self.assertEqual(target_restored.shape, target_standardized.shape)
        np.testing.assert_allclose(target_restored, 20.0)

        with self.assertRaisesRegex(ValueError, 'unsupported width'):
            ms_loader.inverse_transform(np.zeros((2, 3, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'feature dimension'):
            ms_loader.inverse_transform(np.array(0.0))

        m_loader, _ = self.make_generic_loader(feature_type='M')
        with self.assertRaisesRegex(ValueError, 'unsupported width'):
            m_loader.inverse_transform(np.zeros((2, 1), dtype=np.float32))

        s_loader, _ = self.make_generic_loader(feature_type='S')
        s_restored = s_loader.inverse_transform(
            np.zeros((2, 3, 1), dtype=np.float32)
        )
        np.testing.assert_allclose(s_restored, 20.0)

    def test_explicit_train_generator_can_be_saved_and_restored(self):
        generator = torch.Generator().manual_seed(2024)
        loader, _ = self.make_generic_loader(generator=generator)
        train = loader.get_train()

        self.assertIs(train.generator, generator)
        state = loader.get_train_generator_state()
        expected_generator = torch.Generator()
        expected_generator.set_state(state)
        expected = torch.rand(5, generator=expected_generator)

        _ = torch.rand(5, generator=generator)
        loader.set_train_generator_state(state)
        actual = torch.rand(5, generator=loader.train_generator)
        self.assertTrue(torch.equal(actual, expected))
        self.assertIsNone(self.make_generic_loader()[0].get_train_generator_state())

    def test_get_train_can_inject_generator_after_construction(self):
        loader, _ = self.make_generic_loader()
        generator = torch.Generator().manual_seed(7)
        train = loader.get_train(generator=generator)
        self.assertIs(loader.train_generator, generator)
        self.assertIs(train.generator, generator)

    def test_missing_target_is_rejected_for_s_and_ms_but_not_m(self):
        path = self.write_csv(
            'without_target.csv',
            self.generic_rows(30, include_target=False),
            ['date', 'a'],
        )
        for feature_type in ('S', 'MS'):
            with self.subTest(feature_type=feature_type):
                with self.assertRaisesRegex(ValueError, 'target column'):
                    CustomDataLoader(path, 4, 3, 2, feature_type, target='OT')

        with redirect_stdout(io.StringIO()):
            loader = CustomDataLoader(path, 4, 3, 2, 'M', target='OT')
        self.assertEqual(loader.n_feature, 1)

    def test_file_parameter_numeric_and_finite_validation(self):
        with self.assertRaises(FileNotFoundError):
            CustomDataLoader(self.root / 'missing.csv', 4, 3, 2, 'M')

        valid_path = self.write_csv(
            'valid.csv', self.generic_rows(30), ['date', 'a', 'OT']
        )
        with self.assertRaisesRegex(ValueError, 'dataset_id'):
            CustomDataLoader(valid_path, 4, 3, 2, 'M', dataset_id='')
        invalid_parameters = [
            ('batch_size', 0),
            ('seq_len', -1),
            ('pred_len', True),
        ]
        for parameter, value in invalid_parameters:
            kwargs = dict(
                data=valid_path,
                batch_size=4,
                seq_len=3,
                pred_len=2,
                feature_type='M',
            )
            kwargs[parameter] = value
            with self.subTest(parameter=parameter, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    CustomDataLoader(**kwargs)

        non_numeric = self.generic_rows(30)
        non_numeric[5][1] = 'not-a-number'
        non_numeric_path = self.write_csv(
            'non_numeric.csv', non_numeric, ['date', 'a', 'OT']
        )
        with self.assertRaisesRegex(ValueError, 'must all be numeric'):
            CustomDataLoader(non_numeric_path, 4, 3, 2, 'M')

        non_finite = self.generic_rows(30)
        non_finite[5][1] = 'nan'
        non_finite_path = self.write_csv(
            'non_finite.csv', non_finite, ['date', 'a', 'OT']
        )
        with self.assertRaisesRegex(ValueError, 'NaN or infinite'):
            CustomDataLoader(non_finite_path, 4, 3, 2, 'M')

    def test_missing_date_and_empty_split_are_rejected(self):
        path = self.write_csv(
            'no_date.csv',
            [[float(index), float(index * 2)] for index in range(30)],
            ['a', 'OT'],
        )
        with self.assertRaisesRegex(ValueError, "'date' column"):
            CustomDataLoader(path, 4, 3, 2, 'M')

        short_path = self.write_csv(
            'short.csv', self.generic_rows(10), ['date', 'a', 'OT']
        )
        with self.assertRaisesRegex(ValueError, 'empty split'):
            CustomDataLoader(short_path, 2, 4, 3, 'M')

        with self.assertRaisesRegex(ValueError, 'fewer windows than batch_size'):
            CustomDataLoader(
                self.root / 'tiny.csv' if (self.root / 'tiny.csv').exists() else
                self.write_csv(
                    'tiny.csv', self.generic_rows(30), ['date', 'a', 'OT']
                ),
                32,
                3,
                2,
                'M',
            )


if __name__ == '__main__':
    unittest.main()
