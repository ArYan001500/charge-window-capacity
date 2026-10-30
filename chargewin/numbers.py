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
    ("6.3, abstract: UConn propagation, T = 56 d (Ah)", "0.109", C91, "causal_nested_T56/ref_drift"),
    ("6.3: UConn scalar filter, T = 7 d (Ah)", "0.068", C91, "causal_nested_T7/kf_scalar"),
    ("6.3: UConn scalar filter, T = 14 d (Ah)", "0.074", C91, "causal_nested_T14/kf_scalar"),
    ("6.3: UConn bias-aware filter, T = 7 d (Ah)", "0.046", C91, "causal_nested_T7/kf_bias"),
    ("6.3: UConn bias-aware filter, T = 14 d (Ah)", "0.054", C91, "causal_nested_T14/kf_bias"),
    ("6.3: UConn bias-aware filter, T = 28 d (Ah)", "0.061", C91, "causal_nested_T28/kf_bias"),
    ("6.3, abstract: UConn bias-aware filter, T = 56 d (Ah)", "0.062", C91, "causal_nested_T56/kf_bias"),
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
    ("6.4, abstract: rule A worst group, UConn", "0.709", C92, "nested/A/UConn/min_cov"),
    ("6.4, abstract: rule A worst group, KIT", "0.291", C92, "nested/A/KIT/min_cov"),
    ("6.4: rule A worst group, HUST", "0.881", C92, "nested/A/HUST/min_cov"),
    ("6.4, abstract: rule D worst group, UConn", "0.797", C92, "nested/D/UConn/min_cov"),
    ("6.4: rule D worst group, HUST", "0.873", C92, "nested/D/HUST/min_cov"),
    ("6.4, abstract: rule D worst group, KIT", "0.825", C92, "nested/D/KIT/min_cov"),
    ("6.4: rule D mean coverage, UConn", "0.969", C92, "nested/D/UConn/mean_cov"),
    ("6.4: rule D mean coverage, HUST", "0.906", C92, "nested/D/HUST/mean_cov"),
    ("6.4: rule D mean coverage, KIT", "0.973", C92, "nested/D/KIT/mean_cov"),
    ("6.4: rule D per-cell worst group, UConn", "0.801", C92, "nested/D/UConn/min_cell_cov"),
    ("6.4: rule D per-cell worst group, HUST", "0.877", C92, "nested/D/HUST/min_cell_cov"),
    ("6.4: rule D per-cell worst group, KIT", "0.857", C92, "nested/D/KIT/min_cell_cov"),
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
]


def lookup(root, key):
    """Value at `file:path` below `root`; a path step `name[i]` indexes a list."""
    file, path = key.split(":", 1)
    o = json.loads((Path(root) / file).read_text(encoding="utf-8"))
    for part in path.split("/"):
        m = re.fullmatch(r"(.*?)\[(\d+)\]", part)
        o = o[m.group(1)][int(m.group(2))] if m else o[part]
    return float(o)


def rounds_to(value, printed):
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    return abs(value - float(printed)) <= 0.5 * 10 ** -decimals + 1e-9


def check(root):
    failed = []
    print(f"{'paper':<58} {'printed':>8} {'result':>10}", flush=True)
    for where, printed, file, path in PAPER:
        value = lookup(root, f"{file}:{path}")
        ok = rounds_to(value, printed)
        print(f"{where:<58} {printed:>8} {value:>10.4f} {'' if ok else 'FAIL'}", flush=True)
        if not ok:
            failed.append(where)
    if failed:
        raise RuntimeError("numbers differ from the paper: " + "; ".join(failed))
    print(f"all {len(PAPER)} numbers agree with the paper", flush=True)
