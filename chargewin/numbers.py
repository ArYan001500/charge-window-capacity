"""Numbers printed in the paper and the result file each one comes from."""
import json
import re
from pathlib import Path

C11 = "results/C11_UCONN_LOGO_2026-10-23_v1/summary.json"
C14 = "results/C14_HUST_LOGO_2026-10-23_v1/summary.json"
C31 = "results/C31_KIT_LOGO_2026-10-24_usage/summary.json"
C62 = "results/C62_INTEGRAL_REPLICATION_2026-10-24_v1/summary.json"
C78 = "results/C78_TIM_COMPETITORS_2026-10-28_v1/summary.json"
C79 = "results/C79_EXACT_ARCH_2026-10-28_v1/summary.json"
C91 = "results/C91_UCONN_CAUSAL_FUSION_2026-10-29_v1/summary.json"
C92 = "results/C92_NESTED_RULES_AD_2026-10-29_v1/summary.json"
C93 = "results/C93_KIT_HUST_NESTED_FUSION_2026-10-29_v1/summary.json"
C96 = "results/C96_ALL_WINDOW_LAG_2026-10-30_v1/summary.json"
C101 = "results/C101_R4_INTERVAL_DIAG_2026-10-30_v1/summary.json"
C101cd = "results/C101cd_R4_ERRORS_RHO_2026-10-30_v1/summary.json"
C102 = "results/C102_UCONN_MEASURED_TEST_FUSION_2026-10-30_v1/summary.json"
C103 = "results/C103_UCONN_OWN_RATE_2026-10-31_v1/summary.json"
C103b = "results/C103b_KIT_HUST_OWN_RATE_2026-10-31_v1/summary.json"
C104 = "results/C104_GROUP_SIGN_TESTS_2026-10-31_v1/summary.json"
C78b = "results/C78b_CONSTANT_CURRENT_WINDOWS_2026-10-31_v1/summary.json"
C61b = "results/C61b_SEGMENT_CIS_2026-10-31_v1/summary.json"
C96U = "results/C96_ALL_WINDOW_LAG_2026-10-30_v1/summary.json"
C57 = "results/C57_KF_INTERVALS_2026-10-24_v1/summary.json"
C75 = "results/C75_CONSTRAINED_2E_2026-10-24_v1/summary.json"
C06 = "results/C06_CALCE_GENUINE_PARTIAL_2026-10-22_w1800/summary.json"

# (where in the paper, value as printed, file, path inside the JSON)
PAPER = [
    ("Table 3, 6.1: UConn window sensor (Ah)", "0.082", C11, "primary_mean_over_groups/MW_quad"),
    ("Table 3: UConn training mean (Ah)", "0.206", C11, "primary_mean_over_groups/M0"),
    ("Table 3: UConn groups won", "11", C11, "K1_MW_quad_beats_M0"),
    ("Table 3, 6.1: HUST masked window sensor (Ah)", "0.046", C14, "random/primary_mean_over_groups/MW_quad"),
    ("Table 3: HUST training mean (Ah)", "0.072", C14, "random/primary_mean_over_groups/M0"),
    ("6.1: HUST fixed start (Ah)", "0.013", C14, "fixed/primary_mean_over_groups/MW_quad"),
    ("Table 3: KIT window sensor without temperature (Ah)", "0.142", C31, "primary_mean_over_groups/MW_quad"),
    ("Table 3, 6.1: KIT window sensor with temperature (Ah)", "0.103", C31, "primary_mean_over_groups/MW_quadT"),
    ("Table 3: KIT training mean (Ah)", "0.233", C31, "primary_mean_over_groups/M0"),
    ("Table 3: KIT groups won", "13", C31, "K1_MW_quad_beats_M0"),
    ("Table 3: UConn Eq. (2) (Ah)", "0.025", C11, "primary_mean_over_groups/P2"),
    ("Table 3: UConn reference + drift (Ah)", "0.031", C11, "primary_mean_over_groups/P1"),
    ("Table 3: UConn Eq. (2) groups won", "8", C11, "K2_P2_beats_P1"),
    ("Table 3: HUST masked Eq. (2) (Ah)", "0.0036", C14, "random/primary_mean_over_groups/P2"),
    ("Table 3: HUST reference + drift (Ah)", "0.0043", C14, "random/primary_mean_over_groups/P1"),
    ("Table 3: KIT Eq. (2) (Ah)", "0.108", C31, "primary_mean_over_groups/P2"),
    ("Table 3: KIT reference + drift (Ah)", "0.136", C31, "primary_mean_over_groups/P1"),
    ("Table 3: KIT Eq. (2) groups won", "10", C31, "K2_P2_beats_P1"),
    ("Table 3: HUST fixed with descriptors (Ah)", "0.0106", C62, "HUST_fixed/base_F1"),
    ("Table 3: HUST masked with descriptors (Ah)", "0.0440", C62, "HUST_random/base_F1"),
    ("Table 3: KIT with descriptors (Ah)", "0.0973", C62, "KIT/base_F1"),
    ("Table B.1: UConn Guo lookup (Ah)", "0.277", C78, "G2_S1_UConn/guo"),
    ("Table B.1: UConn Shao indicator (Ah)", "0.142", C78, "G2_S1_UConn/shao_E"),
    ("Table B.1: KIT Guo lookup (Ah)", "0.209", C78, "G2_S1_KIT/guo"),
    ("Table B.1: KIT Shao indicator (Ah)", "0.237", C78, "G2_S1_KIT/shao_E"),
    ("Table B.1: UConn Shao network (Ah)", "0.103", C79, "shao_exact_UConn/shao_exact"),
    ("6.3: UConn propagation, T = 7 d (Ah)", "0.050", C91, "causal_nested_T7/ref_drift"),
    ("6.3: UConn propagation, T = 14 d (Ah)", "0.061", C91, "causal_nested_T14/ref_drift"),
    ("6.3: UConn propagation, T = 28 d (Ah)", "0.080", C91, "causal_nested_T28/ref_drift"),
    ("6.3: UConn propagation, T = 56 d (Ah)", "0.109", C91, "causal_nested_T56/ref_drift"),
    ("6.3: UConn scalar filter, T = 7 d (Ah)", "0.068", C91, "causal_nested_T7/kf_scalar"),
    ("6.3: UConn scalar filter, T = 14 d (Ah)", "0.074", C91, "causal_nested_T14/kf_scalar"),
    ("6.3: UConn bias-aware filter, T = 7 d (Ah)", "0.046", C91, "causal_nested_T7/kf_bias"),
    ("6.3: UConn bias-aware filter, T = 14 d (Ah)", "0.054", C91, "causal_nested_T14/kf_bias"),
    ("6.3: UConn bias-aware filter, T = 28 d (Ah)", "0.061", C91, "causal_nested_T28/kf_bias"),
    ("6.3: UConn bias-aware filter, T = 56 d (Ah)", "0.062", C91, "causal_nested_T56/kf_bias"),
    ("6.3: groups better than propagation, T = 7 d", "6", C91, "causal_nested_T7/bias_beats_ref_groups"),
    ("6.3: groups better than propagation, T = 56 d", "9", C91, "causal_nested_T56/bias_beats_ref_groups"),
    ("6.3: measured test, propagation, T = 56 d (Ah)", "0.105", C102, "measured_T56/ref_drift"),
    ("6.3: measured test, bias-aware filter, T = 56 d (Ah)", "0.054", C102, "measured_T56/kf_bias"),
    ("6.3: measured test, reduction (%)", "49", C102, "measured_T56/reduction_pct"),
    ("6.3: measured test, groups better", "10", C102, "measured_T56/bias_beats_ref_groups"),
    ("6.3: KIT bias-aware filter, every check-up (Ah)", "0.086", C93, "KIT_k1/bias_kf"),
    ("6.3: KIT window sensor on the same windows (Ah)", "0.105", C93, "KIT_k1/window"),
    ("6.4: rule A mean coverage, UConn", "0.919", C92, "nested/A/UConn/mean_cov"),
    ("6.4: rule A mean coverage, HUST", "0.902", C92, "nested/A/HUST/mean_cov"),
    ("6.4: rule A mean coverage, KIT", "0.842", C92, "nested/A/KIT/mean_cov"),
    ("6.4: rule A worst group, UConn", "0.709", C92, "nested/A/UConn/min_cov"),
    ("6.4: rule A worst group, KIT", "0.291", C92, "nested/A/KIT/min_cov"),
    ("6.4: rule A worst group, HUST", "0.881", C92, "nested/A/HUST/min_cov"),
    ("6.4: rule D worst group, UConn", "0.797", C92, "nested/D/UConn/min_cov"),
    ("6.4: rule D worst group, HUST", "0.873", C92, "nested/D/HUST/min_cov"),
    ("6.4: rule D worst group, KIT", "0.825", C92, "nested/D/KIT/min_cov"),
    ("6.4: rule D mean coverage, UConn", "0.969", C92, "nested/D/UConn/mean_cov"),
    ("6.4: rule D mean coverage, HUST", "0.906", C92, "nested/D/HUST/mean_cov"),
    ("6.4: rule D mean coverage, KIT", "0.973", C92, "nested/D/KIT/mean_cov"),
    ("6.4: rule D mean width, UConn (Ah)", "0.67", C92, "nested/D/UConn/mean_width_Ah"),
    ("6.4: rule D mean width, HUST (Ah)", "0.21", C92, "nested/D/HUST/mean_width_Ah"),
    ("6.4: rule D mean width, KIT (Ah)", "0.81", C92, "nested/D/KIT/mean_width_Ah"),
    ("6.4: rule A mean width, UConn (Ah)", "0.44", C92, "nested/A/UConn/mean_width_Ah"),
    ("6.4: rule A mean width, HUST (Ah)", "0.19", C92, "nested/A/HUST/mean_width_Ah"),
    ("6.4: rule A mean width, KIT (Ah)", "0.36", C92, "nested/A/KIT/mean_width_Ah"),
    ("6.4: worst-group CI, UConn, lower", "0.757", C101, "UConn/worst_D_CI95[0]"),
    ("6.4: worst-group CI, UConn, upper", "0.844", C101, "UConn/worst_D_CI95[1]"),
    ("6.4: width-matched pooled interval, UConn", "0.791", C101, "UConn/scaledA_worst"),
    ("6.4: width-matched pooled interval, HUST", "0.901", C101, "HUST/scaledA_worst"),
    ("6.4: width-matched pooled interval, KIT", "0.905", C101, "KIT/scaledA_worst"),
    ("6.1: error relative to measured capacity, UConn (%)", "5.3", C101cd, "UConn/MAPE_pct_measured"),
    ("6.1: error relative to measured capacity, HUST (%)", "4.4", C101cd, "HUST/MAPE_pct_measured"),
    ("6.1: error relative to measured capacity, KIT (%)", "4.5", C101cd, "KIT/MAPE_pct_measured"),
    ("6.1: below 70% state of health, UConn (%)", "11", C101cd, "UConn/bins/<0.70/MAPE_measured"),
    ("6.1: below 70% state of health, KIT (%)", "9", C101cd, "KIT/bins/<0.70/MAPE_measured"),
    ("6.3: Spearman rho", "0.42", C101cd, "spearman/rho"),
    ("Table A.3: KIT all windows (Ah)", "0.132", C96, "KIT/Q/table[1]/sensor_MAE"),
    ("6.1: KIT within 4 days of the next check-up (Ah)", "0.099", C96, "KIT/Q/table[2]/sensor_MAE"),
    ("6.1: KIT at 14-28 days (Ah)", "0.179", C96, "KIT/Q/table[5]/sensor_MAE"),
    ("6.1: UConn all windows (Ah)", "0.097", C96U, "UConn/Q/table[1]/sensor_MAE"),
    ("6.1: UConn all windows, groups below the mean", "10", C96U, "UConn/Q/table[1]/groups_sensor_beats_M0"),
    ("6.4: rule B worst group, UConn", "0.792", C92, "nested/B/UConn/min_cov"),
    ("6.4: rule B worst group, HUST", "0.887", C92, "nested/B/HUST/min_cov"),
    ("6.4: rule B worst group, KIT", "0.843", C92, "nested/B/KIT/min_cov"),
    ("6.4: rule B width, UConn (%)", "44", C92, "nested/B/UConn/mean_width_pct"),
    ("6.3: UConn own-rate propagation, T = 7 d (Ah)", "0.070", C103, "T7/own_ref"),
    ("6.3: UConn own-rate propagation, T = 14 d (Ah)", "0.057", C103, "T14/own_ref"),
    ("6.3: UConn own-rate propagation, T = 28 d (Ah)", "0.074", C103, "T28/own_ref"),
    ("6.3: UConn own-rate propagation, T = 56 d (Ah)", "0.090", C103, "T56/own_ref"),
    ("6.3: UConn filter vs own rate, T = 56 d, CI lower", "-0.032", C103, "T56/ci_kf_bias_minus_own_ref[0]"),
    ("6.3: UConn filter vs own rate, T = 56 d, CI upper", "-0.023", C103, "T56/ci_kf_bias_minus_own_ref[1]"),
    ("6.3: UConn filter better than own rate, T = 56 d", "9", C103, "T56/kf_bias_beats_own_ref_groups"),
    ("6.3: KIT own-rate propagation, every check-up (Ah)", "0.099", C103b, "KIT_k1/own_ref"),
    ("6.3: KIT own-rate propagation, every 2nd check-up (Ah)", "0.139", C103b, "KIT_k2/own_ref"),
    ("Table A.3: KIT own-rate filter, every check-up (Ah)", "0.081", C103b, "KIT_k1/own_kf"),
    ("Table A.3: KIT own-rate filter, every 2nd check-up (Ah)", "0.091", C103b, "KIT_k2/own_kf"),
    ("6.3: HUST own-rate propagation (Ah)", "0.0004", C103b, "HUST_fixed_k1/own_ref"),
    ("5.2: sign test, UConn Eq. (2)", "0.11", C104, "UConn: Eq. (2) vs reference + drift/p"),
    ("6.3: sign test, UConn fusion, T = 7 d", "0.50", C104, "UConn T=7: bias-aware filter vs population-rate propagation/p"),
    ("6.3: sign test, UConn fusion, T = 56 d", "0.03", C104, "UConn T=56: bias-aware filter vs population-rate propagation/p"),
    ("Table A.3: sign test, KIT Eq. (2)", "0.046", C104, "KIT: Eq. (2) vs reference + drift/p"),
    ("6.2, Table B.1: UConn constant-current windows, ours (Ah)", "0.211", C78b, "UConn/ours"),
    ("6.2: UConn constant-current windows, Guo lookup (Ah)", "0.109", C78b, "UConn/guo"),
    ("Appendix B: KIT constant-current windows, ours (Ah)", "0.101", C78b, "KIT/ours"),
    ("Appendix B: KIT constant-current windows, Guo lookup (Ah)", "0.205", C78b, "KIT/guo"),
    ("Table A.1: C61b low segment, groups better", "9", C61b, "LOW/group_wins"),
    ("Table C.1: C57 rule D worst group, KIT", "0.790", C57, "D_norm_groupmax/KIT/min_cov"),
    ("Table C.1: C57 rule B worst group, KIT", "0.762", C57, "B_groupmax/KIT/min_cov"),
    ("Table C.1: C75 mid segment (Ah)", "0.407", C75, "by_position/MID/MAE"),
    ("3.3: CALCE training mean (Ah)", "0.075", C06, "MAE/Q_train_mean/eqcell"),
]

# derived values: (where, printed, function of the result folder)
DERIVED = [
    ("Abstract: UConn fusion vs own-rate propagation, 56 d (%)", "31",
     lambda r: 100 * (1 - lookup(r, f"{C103}:T56/kf_bias") / lookup(r, f"{C103}:T56/own_ref"))),
    ("Abstract: KIT fusion vs own-rate propagation, low end (%)", "13",
     lambda r: 100 * (1 - lookup(r, f"{C103b}:KIT_k1/kf_bias") / lookup(r, f"{C103b}:KIT_k1/own_ref"))),
    ("Abstract: KIT fusion vs own-rate propagation, high end (%)", "19",
     lambda r: 100 * (1 - lookup(r, f"{C103b}:KIT_k1/own_kf") / lookup(r, f"{C103b}:KIT_k1/own_ref"))),
    ("6.3: UConn fusion vs population rate, 56 d (%)", "43",
     lambda r: 100 * (1 - lookup(r, f"{C91}:causal_nested_T56/kf_bias") / lookup(r, f"{C91}:causal_nested_T56/ref_drift"))),
    ("Abstract: KIT error without temperature (% of 3.0 Ah)", "4.7",
     lambda r: 100 * lookup(r, f"{C31}:primary_mean_over_groups/MW_quad") / 3.0),
    ("Abstract: UConn error (% of 2.25 Ah)", "3.6", lambda r: 100 * lookup(r, f"{C11}:primary_mean_over_groups/MW_quad") / 2.25),
    ("Abstract: KIT worst group, rule D (%)", "82", lambda r: 100 * lookup(r, f"{C92}:nested/D/KIT/min_cov")),
    ("Abstract: UConn worst group, rule D (%)", "79.7", lambda r: 100 * lookup(r, f"{C92}:nested/D/UConn/min_cov")),
    ("Abstract: UConn worst group, rule A (%)", "71", lambda r: 100 * lookup(r, f"{C92}:nested/A/UConn/min_cov")),
    ("Abstract: KIT worst group, rule A (%)", "29", lambda r: 100 * lookup(r, f"{C92}:nested/A/KIT/min_cov")),
]


def lookup(root, key):
    """Value at `file:path` below `root`; a path step `name[i]` indexes a list."""
    file, path = key.split(":", 1)
    o = json.loads((Path(root) / file).read_text(encoding="utf-8"))
    for part in path.split("/"):
        m = re.fullmatch(r"(.*?)\[(\d+)\]", part)
        o = o[m.group(1)][int(m.group(2))] if m else o[part]
    if isinstance(o, str) and "/" in o:
        o = o.split("/")[0]
    return float(o)


def rounds_to(value, printed):
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    return abs(value - float(printed)) <= 0.5 * 10 ** -decimals + 1e-9


def check(root):
    failed = []
    print(f"{'paper':<58} {'printed':>8} {'result':>10}", flush=True)
    rows = [(w, p, (lambda r, f=f, q=q: lookup(r, f"{f}:{q}"))) for w, p, f, q in PAPER] + DERIVED
    for where, printed, fn in rows:
        value = fn(root)
        ok = rounds_to(value, printed)
        print(f"{where:<58} {printed:>8} {value:>10.4f} {'' if ok else 'FAIL'}", flush=True)
        if not ok:
            failed.append(where)
    if failed:
        raise RuntimeError("numbers differ from the paper: " + "; ".join(failed))
    print(f"all {len(rows)} numbers agree with the paper", flush=True)
