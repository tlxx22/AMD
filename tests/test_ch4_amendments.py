import copy
import unittest
import numpy as np
import torch
from ch4.protocol import MODELS,CITIES,dataset_protocol,resolved_optimization,validate_plan
from ch4.optimization import optimizer_and_scheduler,training_loss
from ch4.state import update_best,capture,restore,scheduler_state
from ch4.baselines import NativeSpatial
from ch4.data import GraphWindows


class Amendments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_same_dataset_protocol_patience_and_budget(self):
        self.assertEqual(len(MODELS),18);self.assertNotIn('PatchSTG',MODELS)
        total=0
        for domain in ['UrbanEV',*CITIES,'PEMS04','PEMS08']:
            base=dataset_protocol(domain)
            self.assertEqual((base['max_epochs'],base['patience']),(200,40) if domain.startswith('PEMS') else (50,20))
            for name in MODELS:
                resolved=resolved_optimization(name,domain);resolved.pop('auxiliary');resolved.pop('cap')
                self.assertEqual(resolved,base)
            total+=len(MODELS)*base['max_epochs']*(4 if domain=='UrbanEV' else 1)
        self.assertEqual(total,16200)

    def test_strict_best_equal_does_not_reset(self):
        for patience in (20,40):
            state=dict(best_mse=None,stale=0)
            self.assertEqual(update_best(state,1.,1,patience),(True,False))
            for epoch in range(2,patience+1):self.assertEqual(update_best(state,1.,epoch,patience),(False,False))
            self.assertEqual(update_best(state,1.,patience+1,patience),(False,True))
            self.assertEqual(state['best_epoch'],1)
            self.assertEqual(update_best(state,.9,patience+2,patience),(True,False))
            self.assertEqual(state['stale'],0)

    def test_resolved_216_plan_rejects_conflicting_patience_and_batch(self):
        import json
        from pathlib import Path
        path=Path(__file__).resolve().parents[1]/'ch4/comparison-plan.json'
        plan=json.loads(path.read_text());self.assertTrue(validate_plan(plan))
        changed=copy.deepcopy(plan);changed['tasks'][0]['training']['patience']=15
        with self.assertRaisesRegex(ValueError,'uniform'):validate_plan(changed)
        changed=copy.deepcopy(plan);changed['tasks'][0]['training']['batch']=16;changed['tasks'][0]['training']['eval_batch']=16
        with self.assertRaises(ValueError):validate_plan(changed)
        changed=copy.deepcopy(plan);changed['tasks'][-1]=changed['tasks'][0]
        with self.assertRaisesRegex(ValueError,'coverage'):validate_plan(changed)

    def test_onecycle_fixed_total_and_resume_exact(self):
        torch.manual_seed(2024);model=torch.nn.Linear(2,1)
        cfg=resolved_optimization('DLinear','UrbanEV');opt,sch=optimizer_and_scheduler(model,cfg,2)
        self.assertEqual(sch.total_steps,100);self.assertAlmostEqual(opt.param_groups[0]['lr'],4e-5)
        self.assertAlmostEqual(opt.param_groups[0]['betas'][0],.95)
        generator=torch.Generator().manual_seed(2024);state=dict(epoch=0,updates=0,best_mse=None,stale=0)
        def step(m,o,s,g,st):
            x=torch.rand(3,2,generator=g);o.zero_grad();loss=m(x).square().mean();loss.backward();o.step();s.step();st['updates']+=1
        step(model,opt,sch,generator,state)
        checkpoint=copy.deepcopy(capture(model,opt,sch,generator,state,{'fixture':True},torch.device('cpu')))
        step(model,opt,sch,generator,state)
        expected=copy.deepcopy(model.state_dict());expected_sch=scheduler_state(sch)
        other=torch.nn.Linear(2,1);oo,ss=optimizer_and_scheduler(other,cfg,2);gg=torch.Generator()
        st=restore(checkpoint,other,oo,ss,gg,{'fixture':True},torch.device('cpu'));step(other,oo,ss,gg,st)
        for k in expected:self.assertTrue(torch.equal(expected[k],other.state_dict()[k]))
        self.assertEqual(scheduler_state(ss),expected_sch);self.assertEqual(ss.total_steps,100)

    def test_testam_slots_midnight_auxiliaries_and_labels(self):
        for h in (1,3,6,9,12):
            times=np.arange('2023-01-01T12','2023-01-03T00',dtype='datetime64[h]')
            x=np.zeros((len(times),3,3),np.float32);y=np.zeros((len(times),3),np.float32)
            ds=GraphWindows(x,y,np.ones_like(y,dtype=bool),times,0,len(times),offset=h)
            sample=ds[0];batch={k:v[None] if torch.is_tensor(v) else v for k,v in sample.items()}
            model=NativeSpatial('TESTAM',3,1,3,offset=h).eval()
            captured=[]
            hook=model.model.identity_expert.register_forward_pre_hook(lambda m,a:captured.append(a[0].clone()))
            raw=[]
            handle=model.model.register_forward_hook(lambda m,a,o:raw.append(o))
            # gate_out requests auxiliary tensors without training dropout.
            pred,extra=model(batch,diagnostic_aux=True);hook.remove();handle.remove()
            self.assertEqual(int(captured[1][0,h-1]),h-1)  # t=23:00 -> next day's Hth hour
            self.assertAlmostEqual(float(batch['future_calendar'][0,0,0]),(h-1)/24,places=6)
            self.assertTrue(torch.equal(pred,raw[0][0][...,h-1:h][:,0].permute(0,2,1)))
            self.assertTrue(torch.equal(extra['experts'],raw[0][2][:,:,h-1:h]))
            self.assertTrue(torch.equal(extra['gate'],raw[0][1][:,:,h-1:h]))
            cfg=resolved_optimization('TESTAM','UrbanEV')
            loss,_=training_loss(pred,batch['y'],batch['mask'],extra,cfg,1)
            grad=torch.autograd.grad(loss,raw[0][2],retain_graph=True)[0]
            other=torch.cat((grad[:,:,:h-1],grad[:,:,h:]),2)
            self.assertTrue(torch.equal(other,torch.zeros_like(other)))
            self.assertGreater(float(grad[:,:,h-1:h].abs().sum()),0)
            self.assertEqual(batch['y'].shape[1],1)  # no other true future labels supplied

    def test_pems08_native_structure_and_five_minute_calendar(self):
        times=np.arange('2016-07-01T23:00','2016-07-02T03:00',dtype='datetime64[5m]')
        x=np.ones((len(times),4,3),np.float32);target=np.zeros((len(times),4),np.float32)
        sample=GraphWindows(x,target,np.ones_like(target,dtype=bool),times,0,len(times),prediction=12)[0]
        batch={k:v[None] if torch.is_tensor(v) else v for k,v in sample.items()}
        self.assertEqual(batch['time_marks'].shape[-1],5)
        self.assertEqual(float(batch['future_calendar'][0,0,0]),0.)
        for name in ('HimNet','AGCRN'):
            model=NativeSpatial(name,3,12,4,'PEMS08').eval()
            with torch.no_grad():prediction,_=model(batch)
            self.assertEqual(prediction.shape,(1,12,4));self.assertTrue(torch.isfinite(prediction).all())


if __name__=='__main__':unittest.main()
