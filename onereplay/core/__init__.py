"""Framework-agnostic OneReplay core: regularizer, covariance, and modeling."""

from onereplay.core.covariance import (
    load_covariance_file,
    make_covariance_hook,
    move_covariances_to_device,
    register_covariance_hooks,
    save_covariance_payload,
    to_identity_covariances,
)
"""Framework-agnostic OneReplay core: regularizer, covariance, and modeling.

Heavy optional deps (peft) are imported lazily inside the modeling helpers that
need them, so lightweight callers like the synthetic regularizer checks can
import covariance / regularizer / snapshot helpers without pulling peft in.
"""

from onereplay.core.covariance import (
    load_covariance_file,
    make_covariance_hook,
    move_covariances_to_device,
    register_covariance_hooks,
    save_covariance_payload,
    to_identity_covariances,
)
from onereplay.core.modeling import (
    attach_adapter,
    build_lora_model,
    find_target_linear_module_names,
    is_lora_adapter_dir,
    join_model_path,
    load_causal_lm_and_tokenizer,
    print_trainable_parameters,
    set_seed,
    snapshot_reference_weights,
)
from onereplay.core.regularizer import (
    ReplayRegularizer,
    full_covariance_regularizer,
    get_lora_weight_matrices,
    lookup_covariance,
    lora_covariance_regularizer,
    strip_peft_prefix,
)

__all__ = [
    "ReplayRegularizer",
    "attach_adapter",
    "build_lora_model",
    "find_target_linear_module_names",
    "full_covariance_regularizer",
    "get_lora_weight_matrices",
    "is_lora_adapter_dir",
    "join_model_path",
    "load_causal_lm_and_tokenizer",
    "load_covariance_file",
    "lookup_covariance",
    "lora_covariance_regularizer",
    "make_covariance_hook",
    "move_covariances_to_device",
    "print_trainable_parameters",
    "register_covariance_hooks",
    "save_covariance_payload",
    "set_seed",
    "snapshot_reference_weights",
    "strip_peft_prefix",
    "to_identity_covariances",
]
