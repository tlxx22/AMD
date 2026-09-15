"""Independent S2/THLS composition metadata; no new model mathematics."""
from models.modules import sonnet_mvca_target_residual as sonnet_spec
from models.modules import target_history_local_shape_residual as thls_spec

IMPLEMENTATION_VARIANT = "amd-m4-sonnet-s2-thls-raw-history-v1"
DEVELOPMENT_PROTOCOL = "m4_sonnet_s2_thls_urbanev_two_arm_from_scratch_v1"
CONTROL_ABLATION_ID = "M4_SONNET_THLS_CONTROL"
ABLATION_ID = "M4_SONNET_THLS"
ARCHITECTURE_IDENTITY = "sonnet_s2_then_mdm_ddi_plus_original_revin_target_thls_v1"
HISTORY_SOURCE = "original_amd_revin_target_before_sonnet_clone_no_detach"
INITIALIZATION_POLICY = "matched_amd_sonnet_and_isolated_thls_v1"


def configuration(enabled):
    if type(enabled) is not bool:
        raise TypeError("S/J THLS switch must be bool")
    return {
        "architecture_identity": ARCHITECTURE_IDENTITY,
        "history_source": HISTORY_SOURCE,
        "initialization_policy": INITIALIZATION_POLICY,
        "sonnet_architecture": sonnet_spec.SONNET_ARCHITECTURE_IDENTITY,
        "thls_structure": thls_spec.STRUCTURE_CONTRACT_VERSION,
        "sonnet_enabled": True, "thls_enabled": enabled,
        "run_seed": 2024, "train_generator_seed": 2024,
        "module_init_seed": 2024, "local_shape_init_seed": 2024 if enabled else None,
        "model_form": "train", "state_width": 56,
    }


def module_connection(enabled):
    configuration(enabled)
    return ("X->RevIN(z)->SonnetS2->MDM(U)->DDI; "
            + ("target+=THLS(z_target_pre_S2); " if enabled else "")
            + "AMS(experts=final_hidden,selector=U)")


# Separate comparison identity. Legacy two-arm helpers above are unchanged.
COMPARISON_VARIANT = "amd-m4-sonnet-s2-thls-three-arm-v1"
COMPARISON_PLAN_ID = "m4_sonnet_s2_thls_dual_dataset_three_arm_from_scratch_v1"
COMPARISON_PROTOCOLS = {
    "UrbanEV": "m4_sonnet_s2_thls_urbanev_three_arm_from_scratch_v1",
    "ETTm1": "m4_sonnet_s2_thls_ettm1_full_horizon_three_arm_from_scratch_v1",
}
COMPARISON_ARMS = {"M4_NSJ_N": (False, True), "M4_NSJ_S": (True, False),
                   "M4_NSJ_J": (True, True)}
COMPARISON_INITIALIZATION = "matched_amd_shared_s2_shared_thls_isolated_v1"


def comparison_configuration(dataset, horizon, arm):
    if dataset not in COMPARISON_PROTOCOLS or arm not in COMPARISON_ARMS:
        raise ValueError("NSJ requires an exact dataset/protocol/arm")
    horizons = (3, 6, 9, 12) if dataset == "UrbanEV" else (96, 192, 336, 720)
    if type(horizon) is not int or horizon not in horizons:
        raise ValueError("NSJ horizon mismatch")
    sonnet, thls = COMPARISON_ARMS[arm]
    return {
        "comparison_plan_id": COMPARISON_PLAN_ID,
        "development_protocol_id": COMPARISON_PROTOCOLS[dataset],
        "dataset_id": dataset, "ablation_id": arm, "label_horizon": horizon,
        "architecture_identity": ARCHITECTURE_IDENTITY,
        "history_source": HISTORY_SOURCE, "history_consumed": thls,
        "initialization_policy": COMPARISON_INITIALIZATION,
        "sonnet_architecture": sonnet_spec.SONNET_ARCHITECTURE_IDENTITY,
        "thls_structure": thls_spec.STRUCTURE_CONTRACT_VERSION,
        "sonnet_enabled": sonnet, "sonnet_instantiated": sonnet,
        "thls_enabled": thls, "thls_instantiated": thls,
        "run_seed": 2024, "train_generator_seed": 2024,
        "module_init_seed": 2024 if sonnet else None,
        "local_shape_init_seed": 2024 if thls else None,
        "model_form": "train", "state_width": 56 if dataset == "UrbanEV" else 1056,
    }


def comparison_connection(dataset, horizon, arm):
    conf = comparison_configuration(dataset, horizon, arm)
    return ("X->RevIN(z)->" + ("SonnetS2->" if conf["sonnet_enabled"] else "")
            + "MDM(U)->DDI; " + ("target+=THLS(z_target_pre_S2); " if conf["thls_enabled"] else "")
            + "AMS(experts=final_hidden,selector=U)")
