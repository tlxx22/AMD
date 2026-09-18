"""External synthetic driver of eight bound author-native Models; no model math."""
import importlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest


class SourceTrainability(unittest.TestCase):
    def id(self):
        from restricted_io_guard import require_installed
        return 'm5_model_adapter.SourceTrainability.test_' + require_installed()['source']

    def test_trainability(self):
        from restricted_io_guard import require_installed
        from m5_entry import MANIFEST, read, write, sha
        from resource_budget import INSTANCE
        import numpy as np
        import torch
        c = require_installed()
        spec = read(MANIFEST)['sources'][c['source']]
        out = Path(c['output_root'])
        device = torch.device(c['device'])
        random.seed(2024)
        np.random.seed(2024)
        torch.manual_seed(2024)
        # Keep native complex64 Koopman parameters; never discard imaginary math.
        torch.set_default_dtype(torch.float32)
        sys.path.insert(0, spec['root'])
        module = importlib.import_module(spec['module'])
        self.assertEqual(str(Path(module.__file__).resolve()), spec['entry'])
        def construct():
            cls = getattr(module, spec['class'])
            if c['source'] == 'AMD-upstream':
                model = cls(**spec['config'])
            elif c['source'] == 'Sonnet':
                model = cls(SimpleNamespace(**spec['config']), datamodule=None)
            else:
                model = cls(SimpleNamespace(**spec['config']))
            return model.to(device)
        def optimizer(model):
            # PyTorch 2.0.1 foreach Adam mishandles complex weight decay shapes.
            # Native Sonnet keeps its complex Koopman math; use the equivalent
            # supported single-tensor Adam API only for this compatibility case.
            compat = {'foreach': False} if c['source'] == 'Sonnet' else {}
            return torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-7,
                                    betas=(.9,.999), eps=1e-8, **compat)
        model = construct()
        opt = optimizer(model)
        self.assertTrue(all(p.dtype in (torch.float32, torch.complex64) for p in model.parameters()))
        generator = torch.Generator().manual_seed(2024)
        train_x = torch.randn(28,96,7,generator=generator)
        val_x = torch.randn(8,96,7,generator=generator)
        def labels(x):
            return .6*x[:,-24:,-1:] + .2*x[:,-24:,0:1] + .1*x[:,-1:,-1:]
        train_y, val_y = labels(train_x), labels(val_x)
        steps, nonzero_steps, changed_steps = 0, 0, 0
        losses, validations, shapes = [], [], []
        def finite(value):
            if torch.is_tensor(value):
                self.assertTrue(bool(torch.isfinite(value).all()), 'nonfinite tensor')
            elif isinstance(value, (tuple,list)):
                for item in value: finite(item)
        def predict(x):
            x = x.to(device)
            result = model(x,None,None,None) if spec['four_arg'] else model(x)
            finite(result)
            pred = result[0] if isinstance(result, tuple) else result
            self.assertEqual(tuple(pred.shape[:2]), (4,24))
            if not shapes: shapes.append(list(pred.shape))
            return pred[:,:,-1:]
        def step(x,y):
            nonlocal steps, nonzero_steps, changed_steps
            model.train()
            opt.zero_grad(set_to_none=True)
            pred = predict(x)
            loss = torch.nn.functional.mse_loss(pred, y.to(device))
            finite(loss)
            loss.backward()
            active = []
            for p in model.parameters():
                if p.grad is not None:
                    finite(p.grad)
                    if bool(torch.count_nonzero(p.grad)): active.append(p)
            self.assertTrue(active, 'no nonzero target gradient')
            nonzero_steps += 1
            before = [p.detach().clone() for p in active]
            opt.step()
            steps += 1
            for p in model.parameters(): finite(p)
            self.assertTrue(any(not torch.equal(p,b) for p,b in zip(active,before)), 'no parameter update')
            changed_steps += 1
            losses.append(float(loss.detach()))
            INSTANCE.sample(torch)
            write(out/'progress.json', dict(completed_adam_steps=steps, nonzero_gradient_steps=nonzero_steps,
                                            changed_steps=changed_steps, losses=losses))
        def validation():
            model.eval()
            with torch.no_grad():
                values = [torch.nn.functional.mse_loss(predict(val_x[i:i+4]), val_y[i:i+4].to(device))
                          for i in (0,4)]
            value = float(torch.stack(values).mean())
            finite(torch.tensor(value))
            validations.append(value)
            return value
        def state():
            return dict(model=model.state_dict(), optimizer=opt.state_dict(), steps=steps,
                        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                        manifest_sha256=c['manifest_sha256'], source=c['source'],
                        best_mse=best, best_epoch=best_epoch, purpose=c['purpose'])
        best, best_epoch = float('inf'), None
        for epoch in (1,2):
            for i in range(0,28,4): step(train_x[i:i+4], train_y[i:i+4])
            value = validation()
            if value < best:
                best, best_epoch = value, epoch
                torch.save(state(), out/'best.pt')
        model.eval()
        with torch.no_grad(): reference = predict(val_x[:4]).clone()
        torch.save(state(), out/'last.pt')
        saved = torch.load(out/'last.pt', map_location=device)
        self.assertEqual(saved['steps'],14)
        self.assertEqual(saved['manifest_sha256'],c['manifest_sha256'])
        model = construct()
        opt = optimizer(model)
        model.load_state_dict(saved['model'],strict=True)
        opt.load_state_dict(saved['optimizer'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        model.eval()
        with torch.no_grad(): reproduced = predict(val_x[:4])
        self.assertTrue(torch.equal(reference,reproduced), 'restored eval is not exactly reproducible')
        for i in (0,4): step(train_x[i:i+4],train_y[i:i+4])
        resumed_val = validation()
        if resumed_val < best:
            best, best_epoch = resumed_val, 'resume-2'
            torch.save(state(),out/'best.pt')
        torch.save(state(), out/'resumed-last.pt')
        imported = {}
        for name, mod in list(sys.modules.items()):
            path = getattr(mod,'__file__',None)
            if path and str(Path(path).resolve()).startswith(spec['root'] + '/'):
                path = str(Path(path).resolve())
                self.assertIn(path, spec['files'], 'unbound native source import')
                self.assertEqual(sha(path), spec['files'][path])
                imported[name] = {'path':path,'sha256':spec['files'][path]}
        reserved = torch.cuda.max_memory_reserved() if device.type=='cuda' else 0
        self.assertLessEqual(reserved, 4*1024**3)
        self.assertEqual(steps,16)
        write(out/'trainability.json',dict(status='Passed',device=str(device),steps=steps,
              finite=True,nonzero_gradient_steps=nonzero_steps,actual_update_steps=changed_steps,
              validation_mse=validations,best_mse=best,best_epoch=best_epoch,
              save_restore_eval_exact=True,continued_steps=2,output_shapes=shapes,
              cuda_reserved_peak=reserved,parameter_count=sum(p.numel() for p in model.parameters()),
              native_model_chain=True,native_cli=False,full_runner=False,imported_sources=imported,
              real_dtype='float32',native_complex_parameters=any(p.is_complex() for p in model.parameters()),
              loss='synthetic last-channel MSE; native auxiliary outputs checked finite, not added'))
