"""CPU cotangent layout regression for the explicit shared DDI compatibility fix."""
import json
import math
import unittest

import torch
from models.common import (
    DDI,
    DDI_CPU_GELU_COMPATIBILITY_ID,
    register_ddi_cpu_gelu_compatibility,
)


def _independent_gelu_vjp_reference(x, cotangent):
    values = [g * (.5 * (1 + math.erf(v / math.sqrt(2)))
                  + v * math.exp(-v * v / 2) / math.sqrt(2 * math.pi))
              for v, g in zip(x.detach().reshape(-1).tolist(), cotangent.reshape(-1).tolist())]
    return torch.tensor(values, dtype=torch.float64).reshape_as(x)


class DDIGELUBackwardLayoutTests(unittest.TestCase):
    def test_cpu_cotangent_contiguous_production_boundary_and_analytic_reference(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            ddi = DDI((512, 7), dropout=.1, patch=16, alpha=.5, layernorm=True).eval()
            activations = [m for m in ddi.modules() if isinstance(m, torch.nn.GELU)]
            self.assertEqual(len(activations), 2)
            initial = {k:v.clone() for k,v in ddi.state_dict().items()}
            rng = torch.get_rng_state().clone()
            for module in activations:
                self.assertEqual(module.approximate, 'none')
                self.assertEqual(len(module._forward_hooks), 1)
                register_ddi_cpu_gelu_compatibility(module)
                self.assertEqual(len(module._forward_hooks), 1)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            repaired = activations[-1]
            native = torch.nn.GELU(approximate='none').eval()
            self.assertEqual(len(native._forward_hooks), 0)
            x = torch.linspace(-.6835061311721802, .6835061311721802, 224).reshape(2,16,7).requires_grad_()
            xn = x.detach().clone().requires_grad_()
            g = torch.linspace(-.00020874293113593012, .00020874293113593012,224).reshape(2,7,16).transpose(1,2)
            original = g.clone(); original_stride = g.stride()
            self.assertEqual(x.stride(), (112,7,1)); self.assertEqual(g.stride(), (112,1,16))
            output = repaired(x); reference_output = native(xn)
            self.assertTrue(torch.equal(output, reference_output))
            self.assertEqual(output.stride(), reference_output.stride())
            native_backward_input = {}
            def observe(gout):
                value = gout[0]
                native_backward_input.update(value=value.detach().clone(), stride=value.stride(),
                    contiguous=value.is_contiguous(), shape=value.shape, dtype=value.dtype, device=value.device)
                return None
            handle = output.grad_fn.register_prehook(observe)
            try:
                (gradient,) = torch.autograd.grad(output,x,grad_outputs=g)
            finally:
                handle.remove()
            (reference_gradient,) = torch.autograd.grad(reference_output,xn,grad_outputs=g.contiguous())
            expected = _independent_gelu_vjp_reference(x,g)
            finite = bool(torch.isfinite(gradient).all() and torch.isfinite(reference_gradient).all())
            difference = gradient.double()-expected
            bound = (.5*(1+math.erf(1))+math.exp(-1)/math.sqrt(math.pi))*float(g.abs().max())
            record = dict(compatibility_id=DDI_CPU_GELU_COMPATIBILITY_ID, finite=finite,
                safe_paths_bitwise=torch.equal(gradient,reference_gradient),
                safe_paths_max_abs=float((gradient.double()-reference_gradient.double()).abs().max()) if finite else None,
                analytic_max_abs=float(difference.abs().max()) if finite else None,
                gradient_max_abs=float(gradient.abs().max()) if finite else None, analytic_bound=bound,
                native_backward_cotangent_contiguous=native_backward_input['contiguous'])
            print('DDI_GELU_LAYOUT_RESULT '+json.dumps(record,allow_nan=False),flush=True)
            self.assertTrue(finite); self.assertTrue(torch.equal(gradient,reference_gradient))
            self.assertTrue(torch.allclose(gradient.double(),expected,atol=1e-10,rtol=1e-5))
            self.assertLessEqual(float(gradient.abs().max()),bound+1e-10)
            self.assertTrue(native_backward_input['contiguous'])
            self.assertTrue(torch.equal(native_backward_input['value'],g))
            self.assertEqual(native_backward_input['shape'],g.shape)
            self.assertEqual(native_backward_input['dtype'],g.dtype)
            self.assertEqual(native_backward_input['device'],g.device)
            self.assertTrue(torch.equal(g,original)); self.assertEqual(g.stride(),original_stride)
            with torch.no_grad(): no_grad_output = repaired(x)
            with torch.inference_mode(): inference_output = repaired(x)
            repeated = repaired(x)
            self.assertFalse(no_grad_output.requires_grad); self.assertFalse(inference_output.requires_grad)
            self.assertFalse(no_grad_output._backward_hooks); self.assertFalse(inference_output._backward_hooks)
            self.assertEqual(len(repeated._backward_hooks),1)
            self.assertEqual(len(output._backward_hooks),1)
            self.assertTrue(torch.equal(repeated,output))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(initial.keys(),ddi.state_dict().keys())
            for k,v in initial.items(): self.assertTrue(torch.equal(v,ddi.state_dict()[k]),k)
        finally:
            torch.set_num_threads(previous_threads)
