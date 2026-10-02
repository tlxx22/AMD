"""Scheduler-only CPU fixtures: optimizer events are stubs, real Adam updates=0."""
import copy,inspect,json,types,unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from utils import ch3_onecycle as o
from utils.ch3_baseline_unified_tasks import AUTHOR_RECIPE
from utils.ch3_native_recovery_records import bound


def optimizer():
    obj=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=1e-7)
    def no_update(self):self.synthetic_events=getattr(self,'synthetic_events',0)+1
    obj.step=types.MethodType(no_update,obj)
    return obj


class OneCycleTests(unittest.TestCase):
    def setUp(self):
        self.deny=patch.multiple(torch.cuda,_lazy_init=unittest.mock.Mock(side_effect=AssertionError('GPU forbidden')))
        self.deny.start();self.addCleanup(self.deny.stop)
        self.no_models=patch.object(torch.nn.Module,'__init__',side_effect=AssertionError('model construction forbidden'))
        self.no_models.start();self.addCleanup(self.no_models.stop)

    def test_actual_torch_defaults(self):
        evidence=bound(AUTHOR_RECIPE)
        defaults={k:p.default for k,p in inspect.signature(torch.optim.lr_scheduler.OneCycleLR).parameters.items() if p.default is not inspect.Parameter.empty}
        self.assertEqual(torch.__version__,'2.0.1')
        self.assertEqual(defaults,evidence['signature_defaults'])
        self.assertEqual(o.RECIPE['max_lr'],.01);self.assertEqual(o.RECIPE['pct_start'],.2)

    def test_exact_author_recipe_lr_beta_trace(self):
        ours=optimizer();author=optimizer();cfg=o.configuration(10,10);s=o.Schedule(ours,cfg)
        a=torch.optim.lr_scheduler.OneCycleLR(author,steps_per_epoch=10,pct_start=.2,epochs=10,max_lr=.01)
        rows=[]
        for i in range(100):
            rows.append(ours.param_groups[0]['lr'])
            self.assertEqual(ours.param_groups[0]['lr'],author.param_groups[0]['lr'])
            self.assertEqual(ours.param_groups[0]['betas'],author.param_groups[0]['betas'])
            ours.step();author.step();s.after_successful_update();a.step()
        self.assertEqual(rows[0],.01+(.01/25-.01));self.assertEqual(max(rows),.01);self.assertAlmostEqual(rows[-1],.01/25/10000,places=14)
        self.assertEqual(s.updates,100);self.assertEqual(s.scheduler.last_epoch,100)

    def test_resume_trace_and_beta_identical(self):
        a=optimizer();full=o.Schedule(a,o.configuration(10,10));b=optimizer();part=o.Schedule(b,o.configuration(10,10))
        for i in range(37):a.step();full.after_successful_update();b.step();part.after_successful_update()
        state=copy.deepcopy(part.state_dict());opt_state=copy.deepcopy(b.state_dict())
        c=optimizer();restored=o.Schedule(c,o.configuration(10,10));c.load_state_dict(opt_state);restored.load_state_dict(state,37)
        for i in range(37,100):
            self.assertEqual(full.state_dict(),restored.state_dict());a.step();full.after_successful_update();c.step();restored.after_successful_update()
        self.assertEqual(full.state_dict(),restored.state_dict())

    def test_wrong_scheduler_profile_refused(self):
        s=o.Schedule(optimizer(),o.configuration(10,10));v=s.state_dict();v['config']['max_lr']=.02
        with self.assertRaises(ValueError):s.load_state_dict(v,0)

    def test_wrong_scheduler_phase_refused(self):
        s=o.Schedule(optimizer(),o.configuration(10,10));v=s.state_dict();v['scheduler']['_schedule_phases'][0]['end_step']=99
        with self.assertRaises(ValueError):s.load_state_dict(v,0)

    def test_wrong_optimizer_lr_beta_refused(self):
        s=o.Schedule(optimizer(),o.configuration(10,10));s.optimizer.param_groups[0]['betas']=(.1,.999)
        with self.assertRaises(ValueError):s.validate()

    def test_step_hard_cap(self):
        s=o.Schedule(optimizer(),o.configuration(2,10))
        for _ in range(20):s.optimizer.step();s.after_successful_update()
        with self.assertRaises(ValueError):s.after_successful_update()

    def test_dynamic_groups_only_lr_beta1(self):
        original=[dict(lr=.01,betas=[.9,.999],eps=1e-8,params=['x'],weight_decay=1e-7)]
        changed=copy.deepcopy(original);changed[0].update(lr=.005,betas=[.85,.999])
        self.assertEqual(o.optimizer_groups_static(original),o.optimizer_groups_static(changed))
        changed[0]['betas'][1]=.95
        self.assertNotEqual(o.optimizer_groups_static(original),o.optimizer_groups_static(changed))

    def test_fixed_lr_legacy_has_no_scheduler(self):
        self.assertIsNone(o.build(object(),dict(training=dict(scheduler=None))))

    def test_checkpoint_best_last_scheduler_resume(self):
        import ch3_runner as r
        from utils.ch3_contract import BestState
        from utils.ch3_baseline_unified_tasks import PACKAGE
        class StateContainer:
            def state_dict(self):return {'fixture':torch.zeros(1)}
            def load_state_dict(self,state,strict=True):self.loaded=state
        model=StateContainer();opt=optimizer();s=o.Schedule(opt,o.configuration(10,10));gen=torch.Generator().manual_seed(2024)
        for _ in range(10):opt.step();s.after_successful_update()
        with tempfile.TemporaryDirectory(dir=PACKAGE,prefix='fixture-checkpoint-') as root:
            for name in ('best.pt','last.pt'):
                path=Path(root)/name;r.save_state(path,model,opt,{'profile':'unified-fixture'},BestState(),1,10,gen,s)
                state=torch.load(path,map_location='cpu');self.assertEqual(state['schema'],'ch3-state-v2-onecycle');self.assertEqual(state['scheduler'],s.state_dict())
            other=optimizer();restored=o.Schedule(other,o.configuration(10,10))
            best,epoch,steps=r.restore_state(Path(root)/'last.pt',StateContainer(),other,{'profile':'unified-fixture'},torch.Generator(),restored)
            self.assertEqual((epoch,steps),(1,10));self.assertEqual(restored.state_dict(),s.state_dict())
            with self.assertRaises(ValueError):r.restore_state(Path(root)/'last.pt',StateContainer(),other,{'profile':'different'},torch.Generator(),restored)

    def test_legacy_checkpoint_compatible_new_resume_rejects(self):
        import ch3_runner as r
        from utils.ch3_contract import BestState
        from utils.ch3_baseline_unified_tasks import PACKAGE
        class StateContainer:
            def state_dict(self):return {}
            def load_state_dict(self,state,strict=True):pass
        with tempfile.TemporaryDirectory(dir=PACKAGE,prefix='fixture-checkpoint-') as root:
            path=Path(root)/'last.pt';opt=optimizer();g=torch.Generator()
            r.save_state(path,StateContainer(),opt,{'protocol':'old'},BestState(),1,10,g)
            self.assertEqual(r.restore_state(path,StateContainer(),opt,{'protocol':'old'},g)[1:],(1,10))
            new=o.Schedule(optimizer(),o.configuration(10,10))
            with self.assertRaises(ValueError):r.restore_state(path,StateContainer(),new.optimizer,{'protocol':'new'},g,new)


if __name__=='__main__':unittest.main()
