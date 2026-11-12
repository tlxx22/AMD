"""Targeted CPU-only regression for real resolver flags and exited PIDs."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from m6_ch3_fullcoverage_v10 import reporting as rp,offline_completion as oc
from m6_ch3_fullcoverage_v10.contract import write_new,sha


class OfflineRepair(unittest.TestCase):
    def test_real_resolver_duplicate_flags_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);old=p/'producer';old.mkdir();write_new(old/'runtime.json',dict(synthetic=True));(old/'receipts').mkdir();(old/'C-H6-UV').mkdir()
            write_new(old/'receipts/C-H6-UV.json',dict(synthetic=True));write_new(old/'C-H6-UV/source.json',dict(producer='original'))
            write_new(old/'C-H6-UV/result.json',dict(synthetic=True));(old/'C-H6-UV/predictions.npz').write_bytes(b'synthetic-not-read')
            inherited={'P-NP-AMD':dict(inherited=False,new_forward=0,producer='upstream-new'),'C-H6-UV':dict(inherited=True,new_forward=0,producer='upstream-reused')}
            s=dict(rows={'P-NP-AMD':{},'C-H6-UV':{}},inherited=inherited);before=copy.deepcopy(s)
            sp=p/'sources.json';write_new(sp,s);c=dict(sources_ref=dict(path=str(sp),sha256=sha(sp)),jobs=[dict(id='C-H6-UV',cell='C-H6-UV'),dict(id='O')])
            # External receipt checking is stubbed; the previously omitted
            # real resolve_inputs control/field construction is exercised.
            with patch.object(rp,'verify_sources'),patch('m6_ch3_fullcoverage_v10.queue.checked',return_value={}):out=rp.resolve_inputs(c,old)
            self.assertEqual(len(out),2);self.assertEqual(len(out['C-H6-UV']),2)
            for cell in inherited:
                x=out[cell][0];self.assertIs(x['inherited'],True);self.assertEqual(x['new_forward'],0)
                self.assertEqual(x['upstream_reference_flags'],{k:inherited[cell][k] for k in ('inherited','new_forward')})
            self.assertEqual(s,before);self.assertFalse(out['C-H6-UV'][1]['inherited'])
            self.assertEqual(out['C-H6-UV'][1]['runtime_ref']['path'],str(old/'runtime.json'))
            with patch.object(rp,'verify_sources'),patch('m6_ch3_fullcoverage_v10.queue.checked',return_value={}):
                bad=dict(c,jobs=[dict(id='C-H6-UV',cell='unexpected'),dict(id='O')])
                with self.assertRaises(ValueError):rp.resolve_inputs(bad,old)

    def test_exited_reused_and_live_PID(self):
        x=dict(pid=123,start='456')
        with patch.object(oc,'proc_start',side_effect=FileNotFoundError):self.assertFalse(oc.owned_alive(x))
        with patch.object(oc,'proc_start',return_value='457'):self.assertFalse(oc.owned_alive(x))
        with patch.object(oc,'proc_start',return_value='456'):self.assertTrue(oc.owned_alive(x))
        with patch.object(oc,'proc_start',side_effect=PermissionError):
            with self.assertRaises(PermissionError):oc.owned_alive(x)

    def test_payload_mutation_membership_and_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);data=p/'data';data.mkdir();f=data/'value.json';write_new(f,dict(synthetic=True))
            r=dict(files={'data/value.json':sha(f)});oc.verify_payload(p,r)
            with self.assertRaises(FileExistsError):data.mkdir(exist_ok=False)
            with self.assertRaises(FileExistsError):write_new(f,dict(synthetic=False))
            write_new(data/'extra.json',dict(synthetic=True))
            with self.assertRaises(ValueError):oc.verify_payload(p,r)
            r['files']['data/extra.json']=sha(data/'extra.json');r['files']['data/value.json']='0'*64
            with self.assertRaises(ValueError):oc.verify_payload(p,r)


if __name__=='__main__':unittest.main()
