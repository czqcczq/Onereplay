"""Lightweight unit checks for the full fine-tuning regularizer path.

These mirror the synthetic half of verify_full_regularizer and are meant to
run in CI / local without a GPU or model checkpoint.

    python -m unittest test_code.check_full_regularizer
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from onereplay.core.modeling import snapshot_reference_weights
from onereplay.core.regularizer import (
    ReplayRegularizer,
    full_covariance_regularizer,
    lora_covariance_regularizer,
)


LAYER_SHAPES = ((32, 32), (48, 32))
RANK = 4
ALPHA = 8


class FakeLoraLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, RANK, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(RANK, d_out, bias=False)})
        self.scaling = {"default": float(ALPHA) / float(RANK)}

    def delta_weight(self) -> torch.Tensor:
        return self.scaling["default"] * (
            self.lora_B["default"].weight @ self.lora_A["default"].weight
        )


class LoraStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeLoraLinear(d_in, d_out) for d_in, d_out in LAYER_SHAPES]
        )


class PlainStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(d_in, d_out, bias=False) for d_in, d_out in LAYER_SHAPES]
        )


def make_spd(dim: int, generator: torch.Generator) -> torch.Tensor:
    samples = torch.randn(dim * 2, dim, generator=generator)
    return samples.T @ samples / (dim * 2)


class FullRegularizerTests(unittest.TestCase):
    def test_lora_and_full_paths_agree(self) -> None:
        generator = torch.Generator().manual_seed(0)
        covariances = {
            f"layers.{index}": make_spd(d_in, generator)
            for index, (d_in, _) in enumerate(LAYER_SHAPES)
        }
        lora_stack = LoraStack()
        plain_stack = PlainStack()
        references = {}
        with torch.no_grad():
            for index, layer in enumerate(lora_stack.layers):
                layer.lora_A["default"].weight.normal_(generator=generator)
                layer.lora_B["default"].weight.normal_(generator=generator)
                name = f"layers.{index}"
                base = plain_stack.layers[index].weight
                references[name] = base.detach().clone()
                base.add_(layer.delta_weight())

        lora_reg, _ = lora_covariance_regularizer(lora_stack, covariances)
        full_reg, _ = full_covariance_regularizer(plain_stack, covariances, references)
        rel = abs(float(lora_reg) - float(full_reg)) / max(abs(float(lora_reg)), 1e-12)
        self.assertLess(rel, 1e-5)

    def test_identity_is_frobenius(self) -> None:
        generator = torch.Generator().manual_seed(1)
        covariances = {
            f"layers.{index}": torch.eye(d_in) for index, (d_in, _) in enumerate(LAYER_SHAPES)
        }
        plain_stack = PlainStack()
        references = {}
        expected = 0.0
        with torch.no_grad():
            for index, layer in enumerate(plain_stack.layers):
                name = f"layers.{index}"
                references[name] = layer.weight.detach().clone()
                layer.weight.normal_(generator=generator)
                expected += float((layer.weight - references[name]).pow(2).sum())
        expected /= len(LAYER_SHAPES)
        full_reg, _ = full_covariance_regularizer(plain_stack, covariances, references)
        self.assertAlmostEqual(float(full_reg), expected, places=5)

    def test_zero_at_initialization(self) -> None:
        generator = torch.Generator().manual_seed(2)
        covariances = {
            f"layers.{index}": make_spd(d_in, generator)
            for index, (d_in, _) in enumerate(LAYER_SHAPES)
        }
        plain_stack = PlainStack()
        references = snapshot_reference_weights(plain_stack, covariances)
        full_reg, stats = full_covariance_regularizer(plain_stack, covariances, references)
        self.assertEqual(float(full_reg), 0.0)
        self.assertEqual(stats["used_layers"], len(LAYER_SHAPES))

    def test_tied_weights_snapshotted_once(self) -> None:
        dim = 16
        generator = torch.Generator().manual_seed(3)
        covariances = {
            "first": make_spd(dim, generator),
            "second": make_spd(dim, generator),
        }

        class TiedPair(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = nn.Linear(dim, dim, bias=False)
                self.second = nn.Linear(dim, dim, bias=False)
                self.second.weight = self.first.weight

        model = TiedPair()
        references = snapshot_reference_weights(model, covariances)
        self.assertEqual(len(references), 1)

    def test_regularizer_dispatches_on_reference_weights(self) -> None:
        generator = torch.Generator().manual_seed(4)
        covariances = {
            f"layers.{index}": make_spd(d_in, generator)
            for index, (d_in, _) in enumerate(LAYER_SHAPES)
        }
        plain_stack = PlainStack()
        references = snapshot_reference_weights(plain_stack, covariances)
        with torch.no_grad():
            for layer in plain_stack.layers:
                layer.weight.add_(0.1)

        regularizer = ReplayRegularizer(covariances, reference_weights=references)
        reg, stats = regularizer(plain_stack)
        self.assertGreater(float(reg), 0.0)
        self.assertEqual(stats["used_layers"], len(LAYER_SHAPES))
        self.assertGreater(regularizer.reference_memory_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
