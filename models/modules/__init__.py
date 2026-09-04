"""Reusable thesis model components."""

from models.modules.cross_correlation_embedding import (
    CCE_INSERTION_POINT,
    CCE_SOURCE_IMPORT_CONTRACT_VERSION,
    EARLY_CCE_ARCHITECTURE,
    EARLY_CCE_INPUT_REPRESENTATION,
    FEATURE_SCHEMA_ORDER,
    IDENTITY_RESIDUAL_DELTA_V1,
    LEGACY_WIDTH_COMPATIBILITY_ZERO,
    LATE_CCE_ARCHITECTURE,
    LATE_CCE_INPUT_REPRESENTATION,
    LATE_CCE_INSERTION_POINT,
    ORDERED_AUX_THEN_TARGET,
    REVIN_REUSE_NO_INTERNAL_NORMALIZATION,
    SIGMOID_LOGIT_PLUS_RHO,
    ZERO_SAME,
    CrossCorrelationEmbedding,
)
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
from models.modules.selective_patch_target_exogenous_bridge import (
    GLOBAL_PREDICTION_ROLE_STATE_ONLY,
    PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION,
    PATCH_GATE_ACTIVATION_TWO_SIGMOID,
    PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
    PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
    SELECTIVE_PATCH_V1,
    SelectivePatchTargetExogenousBridge,
)
from models.modules.sonnet_mvca_target_residual import (
    PaperDefinedMVCA,
    SonnetMVCATargetResidual,
)
from models.modules.target_exogenous_bridge import (
    PARALLEL_MULTIVARIATE,
    TARGET_EXOGENOUS,
    TargetExogenousBridge,
)

__all__ = [
    "CCE_INSERTION_POINT",
    "CCE_SOURCE_IMPORT_CONTRACT_VERSION",
    "EARLY_CCE_ARCHITECTURE",
    "EARLY_CCE_INPUT_REPRESENTATION",
    "FEATURE_SCHEMA_ORDER",
    "IDENTITY_RESIDUAL_DELTA_V1",
    "LEGACY_WIDTH_COMPATIBILITY_ZERO",
    "LATE_CCE_ARCHITECTURE",
    "LATE_CCE_INPUT_REPRESENTATION",
    "LATE_CCE_INSERTION_POINT",
    "ORDERED_AUX_THEN_TARGET",
    "REVIN_REUSE_NO_INTERNAL_NORMALIZATION",
    "SIGMOID_LOGIT_PLUS_RHO",
    "ZERO_SAME",
    "CrossCorrelationEmbedding",
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
    "GLOBAL_PREDICTION_ROLE_STATE_ONLY",
    "PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION",
    "PATCH_GATE_ACTIVATION_TWO_SIGMOID",
    "PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY",
    "PATCH_GATE_INPUT_QUERY_AND_ATTENTION",
    "SELECTIVE_PATCH_V1",
    "SelectivePatchTargetExogenousBridge",
    "PaperDefinedMVCA",
    "SonnetMVCATargetResidual",
    "PARALLEL_MULTIVARIATE",
    "TARGET_EXOGENOUS",
    "TargetExogenousBridge",
]
