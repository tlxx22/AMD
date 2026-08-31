"""Reusable thesis model components."""

from models.modules.global_mediated_patch_target_exogenous_bridge import (
    GLOBAL_GATE_IDENTITY_INIT,
    GLOBAL_GATE_INPUT_CONTRACT,
    GLOBAL_GATE_SCALAR_PER_PATCH,
    GLOBAL_MEDIATED_PATCH_V1,
    GLOBAL_RESIDUAL_CONTRACT,
    PATCH_ATTENTION_RESIDUAL_NONE,
    GlobalMediatedPatchTargetExogenousBridge,
)
from models.modules.modern_conv_refinement import (
    PeakPreservingModernConvRefinement,
    ReparamLargeKernelDWConv,
)
from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    PATCH_CONDITIONED_V1,
    RIGHT_ZERO_CROP,
    PatchConditionedTargetExogenousBridge,
)
from models.modules.target_exogenous_bridge import (
    PARALLEL_MULTIVARIATE,
    TARGET_EXOGENOUS,
    TargetExogenousBridge,
)

__all__ = [
    "PeakPreservingModernConvRefinement",
    "ReparamLargeKernelDWConv",
    "GLOBAL_GATE_IDENTITY_INIT",
    "GLOBAL_GATE_INPUT_CONTRACT",
    "GLOBAL_GATE_SCALAR_PER_PATCH",
    "GLOBAL_MEDIATED_PATCH_V1",
    "GLOBAL_RESIDUAL_CONTRACT",
    "PATCH_ATTENTION_RESIDUAL_NONE",
    "GlobalMediatedPatchTargetExogenousBridge",
    "FIXED_SINUSOIDAL",
    "PATCH_CONDITIONED_V1",
    "RIGHT_ZERO_CROP",
    "PatchConditionedTargetExogenousBridge",
    "PARALLEL_MULTIVARIATE",
    "TARGET_EXOGENOUS",
    "TargetExogenousBridge",
]
