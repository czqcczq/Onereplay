"""CPU-only check of the OSFT glue in onereplay/core/osft.py.

Needs no GPU and no checkpoint, so it runs anywhere and is the first thing to
try after touching core/osft.py or after moving the mini_trainer clone. It
exercises the parts that do not require a real HuggingFace architecture:

  * the decomposition is lossless -- U S V^T rebuilds the original weight, and
    the factorized forward matches the dense one. This is the property that lets
    an OSFT run start from the same model every other arm starts from.
  * our wrap_optimizer really installs both projections, so the frozen factors
    never move and the trainable ones do.
  * our max_frozen_subspace_leak agrees that the trainable factors stay in the
    orthogonal complement. Computed here in float64 against float64 factors, so
    a genuine leak cannot hide behind dtype noise.
  * our describe_osft rank bookkeeping matches the ratio that was asked for.
  * their prepare_state_dict_for_save, which our save path calls, returns a
    state dict with dense weights and no factors left in it.

What it cannot cover: build_osft_model and save_osft_checkpoint, which need a
real checkpoint. Those are in onereplay/scripts/check_osft.py, to be run on a
node that has one.

Usage: python test_code/check_osft_integration.py

Upstream's logging prints emoji, which a Windows console running the GBK code
page cannot encode. Set PYTHONIOENCODING=utf-8 there. Nothing to do on the
cluster, where the locale is already UTF-8.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from onereplay.core.osft import (  # noqa: E402
    describe_osft,
    frozen_rank_ratio,
    load_upstream,
    max_frozen_subspace_leak,
    parse_target_patterns,
    wrap_optimizer,
)

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


class TinyModel(nn.Module):
    """Two square projections and one rectangular one, in float64.

    float64 so that every tolerance below is about the algorithm rather than
    about bf16's eight mantissa bits. The rectangular layer is there because
    U_high and V_high have different shapes when the matrix is not square, which
    is where an index mistake in a projection would show up.
    """

    def __init__(self, config=None, **_):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)
        self.head = nn.Linear(16, 4, bias=True)
        self.config = config
        self.dtype = torch.float64

    def forward(self, x):
        hidden = self.q_proj(x)
        hidden = self.o_proj(hidden)
        wide = torch.cat([hidden, hidden], dim=-1)
        return self.head(self.down_proj(wide))


def build(unfreeze_rank_ratio: float):
    """Decompose a TinyModel at the given ratio, mirroring our own call order."""

    osft_utils = load_upstream()["osft_utils"]
    torch.manual_seed(0)
    reference = TinyModel().to(torch.float64)

    osft_cls = osft_utils.create_osft_model_class(TinyModel)
    model = osft_cls(None, osft_config={}, initialize_osft=False)
    model = model.to(torch.float64)
    model.load_state_dict(reference.state_dict())
    model.upcast_dtype = torch.float64
    model.output_dtype = torch.float64

    model.osft_config = osft_utils.auto_generate_target_osft_config(
        model,
        target_patterns=["q_proj", "o_proj", "down_proj"],
        rank_ratio=frozen_rank_ratio(unfreeze_rank_ratio),
    )
    model.reinitialize_osft(decompose_existing_weights=True)
    return reference, model


def main() -> int:
    check(
        "parse_target_patterns strips quotes and spaces like upstream's CLI",
        parse_target_patterns("'q_proj', \"v_proj\" ") == ["q_proj", "v_proj"]
        and parse_target_patterns("") is None,
    )
    check(
        "frozen_rank_ratio inverts the unfreeze ratio",
        frozen_rank_ratio(0.25) == 0.75 and frozen_rank_ratio(1.0) == 0.0,
    )

    reference, model = build(unfreeze_rank_ratio=0.25)

    # The head is not in the pattern list, so it must stay an ordinary Linear.
    check(
        "only the targeted matrices were decomposed",
        set(model.osft_config) == {"q_proj.weight", "o_proj.weight", "down_proj.weight"},
        f"{sorted(model.osft_config)}",
    )
    check(
        "the untargeted layer keeps its weight parameter",
        "head.weight" in dict(model.named_parameters()),
    )

    osft_utils = load_upstream()["osft_utils"]
    worst_rebuild = 0.0
    for name in model.osft_config:
        rebuilt = model._reconstruct_weight(
            name, upcast_dtype=torch.float64, output_dtype=torch.float64
        )
        original = dict(reference.named_parameters())[name].double()
        worst_rebuild = max(worst_rebuild, float((rebuilt - original).abs().max()))
    check(
        "decomposition is lossless: U S V^T rebuilds every targeted weight",
        worst_rebuild < 1e-10,
        f"max |dW| = {worst_rebuild:.3g}",
    )

    torch.manual_seed(1)
    probe = torch.randn(3, 16, dtype=torch.float64)
    with torch.no_grad():
        gap = float((reference(probe) - model(probe)).abs().max())
    check(
        "factorized forward matches the dense forward at step 0",
        gap < 1e-10,
        f"max |dy| = {gap:.3g}",
    )

    ranks = describe_osft(model, 0.25)
    # min(16,16)=16 twice and min(32,16)=16 once, frozen ratio 0.75 -> 12 each.
    check(
        "describe_osft reports the ranks the ratio implies",
        ranks["osft_matrices"] == 3
        and ranks["osft_rank_high_total"] == 36
        and ranks["osft_rank_total"] == 48,
        f"{ranks['osft_matrices']} matrices, frozen {ranks['osft_rank_high_total']} "
        f"of {ranks['osft_rank_total']}",
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "osft_" in name
    }
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    wrapped = wrap_optimizer(optimizer, model)
    check("wrap_optimizer returned the same optimizer object", wrapped is optimizer)

    target = torch.randn(3, 4, dtype=torch.float64)
    for _ in range(5):
        optimizer.zero_grad()
        nn.functional.mse_loss(model(probe), target).backward()
        optimizer.step()

    frozen_drift = 0.0
    trainable_drift = 0.0
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        drift = float((parameter.detach() - before[name]).abs().max())
        if "_high" in name:
            frozen_drift = max(frozen_drift, drift)
        else:
            trainable_drift = max(trainable_drift, drift)
    check(
        "high-rank factors did not move across 5 steps",
        frozen_drift == 0.0,
        f"max drift {frozen_drift:.3g}",
    )
    check(
        "low-rank factors did move",
        trainable_drift > 1e-6,
        f"max drift {trainable_drift:.3g}",
    )

    leak = max_frozen_subspace_leak(model)
    check(
        "trainable factors stay orthogonal to the frozen subspace",
        leak < 1e-10,
        f"max overlap {leak:.3g}",
    )

    # An unprojected optimizer must fail the same test, or the check above is
    # only measuring that Adam happened to stay put.
    reference_naive, naive = build(unfreeze_rank_ratio=0.25)
    naive_optimizer = torch.optim.Adam(
        [p for p in naive.parameters() if p.requires_grad], lr=1e-2
    )
    for _ in range(5):
        naive_optimizer.zero_grad()
        nn.functional.mse_loss(naive(probe), target).backward()
        naive_optimizer.step()
    naive_leak = max_frozen_subspace_leak(naive)
    check(
        "the same run without the projections does leak, so the test has teeth",
        naive_leak > 1e-6,
        f"unprojected overlap {naive_leak:.3g} vs projected {leak:.3g}",
    )
    del reference_naive, naive

    state_dict = model.prepare_state_dict_for_save(model.state_dict())
    leftovers = [key for key in state_dict if "osft_" in key]
    check(
        "prepare_state_dict_for_save leaves no factors behind",
        not leftovers,
        f"{leftovers[:3]}" if leftovers else "",
    )
    check(
        "prepare_state_dict_for_save restores the original parameter names",
        all(name in state_dict for name in model.osft_config)
        and state_dict["down_proj.weight"].shape == (16, 32),
    )

    # Ratio 1.0 is the arm's own control: nothing frozen means plain full
    # fine-tuning, which is what makes a 1.0 run comparable with the vanilla one.
    _, wide_open = build(unfreeze_rank_ratio=1.0)
    open_ranks = describe_osft(wide_open, 1.0)
    check(
        "unfreeze_rank_ratio 1.0 freezes no directions",
        open_ranks["osft_rank_high_total"] == 0,
        f"frozen {open_ranks['osft_rank_high_total']}",
    )

    print("\n" + "=" * 68)
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for name in _failures:
            print(f"  {name}")
        return 1
    print("all OSFT glue checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
