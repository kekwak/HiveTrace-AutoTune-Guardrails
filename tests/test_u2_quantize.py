"""Selection of quantized linears; real torchao call is replaced, GPU kernels are checked on the pod."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from u2_quant import quantize


def tiny():
    return nn.Sequential(nn.Embedding(10, 8), nn.Linear(8, 8), nn.LayerNorm(8),
                         nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2)))


class Targets(unittest.TestCase):
    def test_targets_only_linear_and_respect_skip(self):
        self.assertEqual(quantize.linear_targets(tiny(), []), ["1", "3.0", "3.1"])
        self.assertEqual(quantize.linear_targets(tiny(), ["3.0"]), ["1", "3.1"])

    def test_skip_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            quantize.linear_targets(tiny(), ["3.7"])


class Apply(unittest.TestCase):
    def test_none_is_noop(self):
        model = tiny()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        info = quantize.apply(model, "none", [])
        self.assertEqual(info["quantized"], [])
        self.assertEqual(info["method"], "none")
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            quantize.apply(tiny(), "int8", [])

    def test_fp8_calls_quantize_with_filter(self):
        from torchao.quantization import PerRow
        model = tiny()
        with patch("torchao.quantization.quantize_") as call:
            info = quantize.apply(model, "fp8", ["3.1"])
        call.assert_called_once()
        args, kwargs = call.call_args
        self.assertIs(args[0], model)
        config = args[1]
        self.assertEqual(type(config).__name__, "Float8DynamicActivationFloat8WeightConfig")
        granularity = config.granularity if isinstance(config.granularity, (list, tuple)) else [config.granularity]
        self.assertTrue(granularity and all(isinstance(g, PerRow) for g in granularity), granularity)
        # нулевые строки паддинга (gliner, батч > 1) дают масштаб 0 и NaN без нижней границы
        self.assertIsNotNone(config.activation_value_lb)
        self.assertGreater(config.activation_value_lb, 0)
        self.assertLess(config.activation_value_lb, 1e-6)
        chosen = [name for name, m in model.named_modules() if kwargs["filter_fn"](m, name)]
        self.assertEqual(chosen, ["1", "3.0"])
        self.assertEqual(info["quantized"], ["1", "3.0"])
        self.assertEqual(info["skipped"], ["3.1"])
        self.assertEqual(info["granularity"], "per_row")
        self.assertEqual(info["torchao"], "0.17.0")
        self.assertEqual(info["activation_value_lb"], config.activation_value_lb)


if __name__ == "__main__":
    unittest.main()
