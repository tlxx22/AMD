"""Finite CPU fixtures for worker, freeze gate, one-test claims and sealing."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from ch4.worker import run
from ch4.queue import freeze_best,claim_test
from ch4.protocol import resolved_optimization
from ch4.imports import sha
from m6_ch3_supplement_v10.contract import write_new,read


class Execution(unittest.TestCase):
    def test_fixture_worker_and_freeze_once(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='m7-cpu-fixture-') as tmp:
            root=Path(tmp);prefix=root/'prefix';full=root/'synthetic-final';out=root/'fixture';out.mkdir()
            rng=np.random.default_rng(2024);x=rng.normal(size=(76,2,3)).astype(np.float32);y=x[:,:,0].copy()
            valid=np.ones_like(y,dtype=bool);valid[30,1]=False;y[30,1]=np.nan;y[40,0]=0.
            for path,stop in [(prefix,52),(full,76)]:
                path.mkdir()
                arrays=dict(x=x[:stop],y=y[:stop],valid=valid[:stop],times=np.arange('2023-01-01','2023-01-05',dtype='datetime64[h]')[:stop],
                            geo=np.array([[0,1],[0,1]]),sem=np.array([[0,1],[0,1]]),target_scale=np.array([2.,3.]))
                for k,v in arrays.items():np.save(path/(k+'.npy'),v,allow_pickle=False)
                write_new(path/'metadata.json',dict(files={p.name:sha(p) for p in path.iterdir()}))
            cfg=resolved_optimization('DLinear','UrbanEV');cfg.update(max_epochs=2,cap=2,batch=8,eval_batch=8,threads=1)
            task=dict(id='fixture',model='DLinear',domain='UrbanEV',H=1,P=1,C=3,N=2,L=12,
                      node_order=['a','b'],endpoints=[28,52,76],training=cfg,data_path=str(prefix),
                      metadata_sha256=sha(prefix/'metadata.json'),steps_per_epoch=2)
            binding={'fixture_only':True};result=run(task,out,'train',lambda:None,binding,device='cpu')
            self.assertEqual((result['epochs'],result['updates'],result['test_accesses']),(2,4,0))
            self.assertFalse((out/'test-result.json').exists())
            write_new(root/'config.json',dict(tasks=[task]))
            with self.assertRaises(FileNotFoundError):claim_test(root,'fixture')
            freeze_best(dict(tasks=[task]),root)
            row=claim_test(root,'fixture')
            with self.assertRaises(FileExistsError):claim_test(root,'fixture')
            final=run(task,out,'test',lambda:None,row,test_folder=full,device='cpu')
            pred=np.load(out/'test_predictions/prediction.npy');truth=np.load(out/'test_predictions/truth.npy')
            mask=np.load(out/'test_predictions/mask.npy');err=pred.astype(float)[mask]-truth.astype(float)[mask]
            self.assertEqual(final['formal_train_standardized']['count'],24)
            self.assertAlmostEqual(final['formal_train_standardized']['mse'],float(np.mean(err**2)),places=12)
            self.assertAlmostEqual(final['formal_train_standardized']['mae'],float(np.mean(abs(err))),places=12)
            self.assertEqual(len(read(root/'frozen-best.json')['tasks']),1)
            self.assertTrue((out/'train-access.jsonl').exists());self.assertTrue((out/'test-access.jsonl').exists())


if __name__=='__main__':unittest.main()
