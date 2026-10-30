"""Analysis steps in run order. Paths are relative to the run folder, which holds data/, tables/ and results/."""

C10 = "tables/C10_UCONN_2026-10-23_v1"
C31D = "tables/C31_KIT_2026-10-24_v1"
C38 = "tables/C38_UCONN_REF_CURVES_2026-10-24_v1"
PARAMS = "data/uconn/cycling_parameters.csv"


def r(name):
    return "results/" + name


C11 = r("C11_UCONN_LOGO_2026-10-23_v1")
C14 = r("C14_HUST_LOGO_2026-10-23_v1")
C15 = r("C15_UCONN_DANN_2026-10-23_v1")
C19 = r("C19_UCONN_PULSE_2026-10-23_v1")
C32 = r("C32_KIT_KALMAN_2026-10-24_v1")
C35 = r("C35_KIT_MLP_PASSIVE_2026-10-24_v1")
C36I = r("C36_UCONN_MLP_PASSIVE_INPUT_2026-10-24")
C49 = r("C49_NAVIDI_FEATURES_2026-10-24_v1")


def step(sid, script, args, out, tier, data=(), section=""):
    return dict(id=sid, script=script, args=list(args), out=out, tier=tier, data=tuple(data), section=section)


# tier "main": the results of Sections 6.1-6.4, Appendix A and the figures (CPU, no neural networks)
# tier "full": neural networks, development studies of Appendix C and the CALCE illustration
STEPS = [
    step("C10a", "c10_uconn_extract.py", ["data/uconn", C10, "cycling"], C10, "main", ["uconn"], "3.2"),
    step("C10b", "c10_uconn_extract.py", ["data/uconn", C10, "rpt"], C10, "main", ["uconn"], "3.2"),
    step("C38", "c38_uconn_full_ref_curves.py", ["data/uconn/rpt_data.zip", C38], C38, "main", ["uconn"], "Fig. 1, 3"),
    step("C11", "c11_uconn_logo.py", [C10, PARAMS, C11], C11, "main", ["uconn"], "6.1"),
    step("C14", "c14_hust_logo.py", ["data/hust/our_data.zip", C14], C14, "main", ["hust"], "6.1"),
    step("C19", "c19_uconn_pulse_complement.py", ["data/uconn/rpt_data.zip", C11, C10, C19], C19, "main", ["uconn"], "5.1"),
    step("C31a", "c31_kit_extract.py", ["data/kit/cell_log_age_30s", "data/kit/result_data", C31D], C31D, "main", ["kit"], "3.2"),
    step("C31", "c31_kit_logo.py", [C31D, r("C31_KIT_LOGO_2026-10-24_usage"), "usage"], r("C31_KIT_LOGO_2026-10-24_usage"), "main", ["kit"], "6.1"),
    step("C31t", "c31_kit_logo.py", [C31D, r("C31_KIT_LOGO_2026-10-24_temp"), "temp"], r("C31_KIT_LOGO_2026-10-24_temp"), "main", ["kit"], "7.4"),
    step("C32", "c32_kit_kalman.py", [C31D, C32], C32, "main", ["kit"], "Appendix C"),
    step("C54", "c54_hust_kalman.py", [C14, r("C54_HUST_KALMAN_2026-10-24_v1")], r("C54_HUST_KALMAN_2026-10-24_v1"), "main", ["hust"], "Table 3"),
    step("C61", "c61_position_integral_features.py", [".", r("C61_POSITION_INTEGRAL_2026-10-24_v1")], r("C61_POSITION_INTEGRAL_2026-10-24_v1"), "main", ["uconn"], "6.1"),
    step("C62", "c62_integral_feature_replication.py", [".", r("C62_INTEGRAL_REPLICATION_2026-10-24_v1")], r("C62_INTEGRAL_REPLICATION_2026-10-24_v1"), "main", ["uconn", "hust", "kit"], "6.1"),
    step("C64", "c64_aging_speed_formula.py", [".", r("C64_AGING_SPEED_2026-10-24_v1")], r("C64_AGING_SPEED_2026-10-24_v1"), "main", ["kit"], "6.3"),
    step("C78", "c78_tim_competitors.py", [".", r("C78_TIM_COMPETITORS_2026-10-28_v1")], r("C78_TIM_COMPETITORS_2026-10-28_v1"), "main", ["uconn", "kit"], "6.2"),
    step("C86", "c86_hust_mlp_sensor.py", ["results", r("C86_HUST_MLP_2026-10-29_v1")], r("C86_HUST_MLP_2026-10-29_v1"), "main", ["hust"], "6.1"),
    step("C87", "c87_taosee_uconn_hust.py", [".", r("C87_TAOSEE_UCONN_HUST_2026-10-29_v1")], r("C87_TAOSEE_UCONN_HUST_2026-10-29_v1"), "main", ["uconn", "hust"], "6.2"),
    step("C90", "c90_nested_calibration.py", [".", r("C90_NESTED_CALIBRATION_2026-10-29_v1")], r("C90_NESTED_CALIBRATION_2026-10-29_v1"), "main", ["uconn", "hust", "kit"], "6.4"),
    step("C91", "c91_uconn_causal_nested_fusion.py", [".", r("C91_UCONN_CAUSAL_FUSION_2026-10-29_v1")], r("C91_UCONN_CAUSAL_FUSION_2026-10-29_v1"), "main", ["uconn"], "6.3"),
    step("C92", "c92_nested_interval_rules.py", [".", r("C92_NESTED_RULES_AD_2026-10-29_v1")], r("C92_NESTED_RULES_AD_2026-10-29_v1"), "main", ["uconn", "hust", "kit"], "6.4"),
    step("C93", "c93_kit_hust_nested_fusion.py", [".", r("C93_KIT_HUST_NESTED_FUSION_2026-10-29_v1")], r("C93_KIT_HUST_NESTED_FUSION_2026-10-29_v1"), "main", ["hust", "kit"], "6.3"),
    step("C95", "c95_uconn_release_time_replay.py", [".", r("C95_UCONN_RELEASE_REPLAY_2026-10-30_v1")], r("C95_UCONN_RELEASE_REPLAY_2026-10-30_v1"), "main", ["uconn"], "6.3"),
    step("C96", "c96_all_window_lag.py", [".", r("C96_ALL_WINDOW_LAG_2026-10-30_v1")], r("C96_ALL_WINDOW_LAG_2026-10-30_v1"), "main", ["uconn", "kit"], "Table A.3"),
    step("C97", "c97_hust_regrouping.py", [".", r("C97_HUST_REGROUPING_2026-10-30_v1")], r("C97_HUST_REGROUPING_2026-10-30_v1"), "main", ["hust"], "Table A.3"),
    step("C99", "c99_cluster_bootstrap_scale.py", [".", r("C99_CLUSTER_BOOT_SCALE_2026-10-30_v1")], r("C99_CLUSTER_BOOT_SCALE_2026-10-30_v1"), "main", ["uconn", "hust", "kit"], "Table A.3"),
    step("C100", "c100_uconn_correlated_noise_filter.py", [".", r("C100_UCONN_AR_NOISE_FILTER_2026-10-30_v1")], r("C100_UCONN_AR_NOISE_FILTER_2026-10-30_v1"), "main", ["uconn"], "Table A.3"),
    step("C101", "c101_interval_diagnostics.py", [".", r("C101_R4_INTERVAL_DIAG_2026-10-30_v1")], r("C101_R4_INTERVAL_DIAG_2026-10-30_v1"), "main", ["uconn", "hust", "kit"], "Table A.3"),
    step("C101cd", "c101_errors_rho.py", [".", r("C101cd_R4_ERRORS_RHO_2026-10-30_v1")], r("C101cd_R4_ERRORS_RHO_2026-10-30_v1"), "main", ["uconn", "hust", "kit"], "Table A.3"),
    step("C102", "c102_uconn_measured_test_fusion.py", [".", r("C102_UCONN_MEASURED_TEST_FUSION_2026-10-30_v1")], r("C102_UCONN_MEASURED_TEST_FUSION_2026-10-30_v1"), "main", ["uconn"], "Table A.3"),
    step("C88", "c88_dataset_figure_data.py", [".", r("C88_DATASET_FIGURE_DATA_2026-10-29")], r("C88_DATASET_FIGURE_DATA_2026-10-29"), "main", ["uconn", "hust", "kit"], "Fig. 1"),
    step("C94", "c94_overview_causal_data.py", [".", "21"], r("C94_OVERVIEW_CAUSAL_2026-10-29"), "main", ["uconn"], "Fig. 3"),
    step("C15", "c15_uconn_dann.py", [C11, C15, C10], C15, "full", ["uconn"], "Appendix C"),
    step("C24", "c24_kalman_fusion.py", [C19, C11, r("C24_KALMAN_FUSION_2026-10-23_v1")], r("C24_KALMAN_FUSION_2026-10-23_v1"), "full", ["uconn"], "Appendix A"),
    step("C25", "c25_bias_aware_kalman.py", [C19, C11, r("C25_BIAS_KF_2026-10-23_v1")], r("C25_BIAS_KF_2026-10-23_v1"), "full", ["uconn"], "Appendix A"),
    step("C35", "c35_kit_mlp_passive.py", [C31D, C32, C35], C35, "full", ["kit"], "6.1"),
    step("C35k", "c32_kit_kalman.py", [C31D, r("C35_KIT_KALMAN_MLP_2026-10-24_v1"), C35], r("C35_KIT_KALMAN_MLP_2026-10-24_v1"), "full", ["kit"], "7.2"),
    step("C36i", "c36_uconn_mlp_input.py", [C19, C15, C36I], C36I, "full", ["uconn"], "7.2"),
    step("C36k", "c25_bias_aware_kalman.py", [C36I, C11, r("C36_UCONN_BIAS_KF_MLP_2026-10-24_v1")], r("C36_UCONN_BIAS_KF_MLP_2026-10-24_v1"), "full", ["uconn"], "7.2"),
    step("C36", "c36_bias_persistence.py", ["results", r("C36_BIAS_PERSISTENCE_2026-10-24_v1")], r("C36_BIAS_PERSISTENCE_2026-10-24_v1"), "full", ["uconn", "kit"], "7.2"),
    step("C49", "c49_navidi_features.py", ["data/uconn/rpt_data.zip", ".", C49], C49, "full", ["uconn"], "7.2"),
    step("C49b", "c49b_navidi_vs_window.py", [".", C49, r("C49b_NAVIDI_VS_WINDOW_2026-10-24_v1")], r("C49b_NAVIDI_VS_WINDOW_2026-10-24_v1"), "full", ["uconn"], "7.2"),
    step("C53", "c53_ci_and_conformal.py", ["results", r("C53_CI_CONFORMAL_2026-10-24_v1")], r("C53_CI_CONFORMAL_2026-10-24_v1"), "full", ["uconn", "hust", "kit"], "Appendix A"),
    step("C55", "c55_kit_taosee_baseline.py", [".", r("C55_KIT_TAOSEE_BASELINE_2026-10-24_v1")], r("C55_KIT_TAOSEE_BASELINE_2026-10-24_v1"), "full", ["kit"], "6.2"),
    step("C56", "c56_shift_robust_calibration.py", ["results", r("C56_SHIFT_CALIBRATION_2026-10-24_v1")], r("C56_SHIFT_CALIBRATION_2026-10-24_v1"), "full", ["uconn", "hust", "kit"], "Appendix A"),
    step("C57", "c57_kf_intervals.py", [".", r("C57_KF_INTERVALS_2026-10-24_v1")], r("C57_KF_INTERVALS_2026-10-24_v1"), "full", ["uconn", "kit"], "Appendix C"),
    step("C58", "c58_kit_tuned_dann.py", [C31D, r("C58_KIT_TUNED_DANN_2026-10-24_v1")], r("C58_KIT_TUNED_DANN_2026-10-24_v1"), "full", ["kit"], "6.2"),
    step("C60", "c60_datasheet_slow_charge.py", [".", r("C60_DATASHEET_SLOW_2026-10-24_v2")], r("C60_DATASHEET_SLOW_2026-10-24_v2"), "full", ["uconn"], "Appendix C"),
    step("C65", "c65_formula_drift_kalman.py", [".", r("C65_FORMULA_DRIFT_KF_2026-10-24_v1")], r("C65_FORMULA_DRIFT_KF_2026-10-24_v1"), "full", ["kit"], "Appendix C"),
    step("C67", "c67_fewer_tests.py", [".", r("C67_FEWER_TESTS_2026-10-24_v1")], r("C67_FEWER_TESTS_2026-10-24_v1"), "full", ["kit"], "Appendix C"),
    step("C68", "c68_charge_irregularity.py", [".", r("C68_CHARGE_IRREGULARITY_2026-10-24_v1")], r("C68_CHARGE_IRREGULARITY_2026-10-24_v1"), "full", ["kit"], "Appendix C"),
    step("C69", "c69_drift_aware_trigger.py", [".", r("C69_DRIFT_TRIGGER_2026-10-24_v1")], r("C69_DRIFT_TRIGGER_2026-10-24_v1"), "full", ["kit"], "Appendix C"),
    step("C70", "c70_uconn_self_aware.py", [".", r("C70_UCONN_SELF_AWARE_2026-10-24_v1")], r("C70_UCONN_SELF_AWARE_2026-10-24_v1"), "full", ["uconn"], "Appendix C"),
    step("C72", "c72_hust_law.py", [".", r("C72_HUST_LAW_2026-10-24_v1")], r("C72_HUST_LAW_2026-10-24_v1"), "full", ["hust"], "Appendix C"),
    step("C74", "c74_two_electrode.py", [".", r("C74_TWO_ELECTRODE_2026-10-24_v1")], r("C74_TWO_ELECTRODE_2026-10-24_v1"), "full", ["uconn"], "Appendix C"),
    step("C75", "c75_constrained_two_electrode.py", [".", r("C75_CONSTRAINED_2E_2026-10-24_v1")], r("C75_CONSTRAINED_2E_2026-10-24_v1"), "full", ["uconn"], "Appendix C"),
    step("C79", "c79_exact_architectures.py", [".", r("C79_EXACT_ARCH_2026-10-28_v1")], r("C79_EXACT_ARCH_2026-10-28_v1"), "full", ["uconn", "kit"], "6.2"),
    step("C80", "c80_hybrid_gate.py", [".", r("C80_HYBRID_GATE_2026-10-28_v1")], r("C80_HYBRID_GATE_2026-10-28_v1"), "full", ["uconn", "kit"], "Appendix C"),
    step("C85", "c85_posthoc_ci.py", [".", r("C85_POSTHOC_CI_2026-10-29_v1")], r("C85_POSTHOC_CI_2026-10-29_v1"), "full", ["uconn", "kit"], "6.2"),
] + [
    step(f"C98_{n}_{a}_{s}", "c98_causal_shao.py", [".", r("C98_CAUSAL_SHAO_2026-10-30_v1"), n, a, str(s)],
         r("C98_CAUSAL_SHAO_2026-10-30_v1"), "full", ["uconn", "kit"], "6.2")
    for n in ("UConn", "KIT") for a in ("causal", "bi") for s in (0, 1, 2)
] + [
    step("C98s", "c98_summarize.py", [".", r("C98_CAUSAL_SHAO_2026-10-30_v1")], r("C98_CAUSAL_SHAO_2026-10-30_v1"), "full", ["uconn", "kit"], "6.2"),
] + [
    step(f"C99b_{s}", "c99_cluster_bootstrap_scale.py", [".", r(f"C99b_CLUSTER_SEEDS_2026-10-30_v1/seed{s}_reps{n}"), str(s), str(n)],
         r(f"C99b_CLUSTER_SEEDS_2026-10-30_v1/seed{s}_reps{n}"), "full", ["uconn", "hust", "kit"], "6.4")
    for s, n in [(101 + i, 30) for i in range(10)] + [(201, 200), (202, 200), (203, 200)]
]
