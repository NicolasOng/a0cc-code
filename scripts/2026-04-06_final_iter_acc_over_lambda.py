'''
Final iteration model accuracy (value & policy heads) over td_lambda.

Loads merged eval series from a set of `output-td<NN>/eval/` directories
(one per lambda) and plots, for each eval dataset, the mean ± 95% CI of
the accuracy at the final training iteration as a function of lambda.

Pass the parent folder containing all the `output-td*/` dirs by editing
OUTPUTS_PARENT_DIR below.
'''
import math

from a0.utils.plotting import Series, load_series, save_series, plot_shaded_error

# ── Edit me ──────────────────────────────────────────────────────────────────
OUTPUTS_PARENT_DIR = "output"  # parent folder containing output-td00/, output-td25/, ...

# (lambda, dirname) — one per run
LAMBDA_DIRS: list[tuple[float, str]] = [
    (0.0,  "output-td00"),
    (0.5,  "output-td50"),
    (0.75, "output-td75"),
    (0.88, "output-td88"),
    (1.0,  "output-td100"),
]

# (label, merged-series filename stem)
SERIES: list[tuple[str, str]] = [
    ("Seen",       "merged_training_gtv_eval"),
    ("Random",     "merged_random_gtv_eval"),
    ("Neighbor 1", "merged_neighbor_1_gtv_eval"),
    ("Neighbor 2", "merged_neighbor_2_gtv_eval"),
    ("Seen NT",       "merged_training_nt_gtv_eval"),
    ("Random NT",     "merged_random_nt_gtv_eval"),
    ("Neighbor 1 NT", "merged_neighbor_1_nt_gtv_eval"),
    ("Neighbor 2 NT", "merged_neighbor_2_nt_gtv_eval"),
]


def build_combined_series(y_key: str) -> Series:
    '''
    Walk every lambda dir and pull the (mean, CI) at the final iteration of
    each merged eval series. Packs everything into a single Series with
        x  = [lambda, ...]
        ys = { "<label>": [mean per lambda, ...],
               "<label>_ci": [ci per lambda, ...] }
    Missing lambdas are NaN-filled so the x-axis stays aligned across labels.
    '''
    series = Series()
    series.x = [lam for lam, _ in LAMBDA_DIRS]  # type: ignore[assignment]
    for label, stem in SERIES:
        means: list[float] = []
        cis: list[float] = []
        for _, dirname in LAMBDA_DIRS:
            path = f"{OUTPUTS_PARENT_DIR}/{dirname}/eval/{stem}.pkl"
            try:
                s = load_series(path)
            except FileNotFoundError:
                print(f"  skip {path} (missing)")
                means.append(math.nan)
                cis.append(math.nan)
                continue
            if not s.x:
                print(f"  skip {path} (empty series)")
                means.append(math.nan)
                cis.append(math.nan)
                continue
            means.append(s.ys[y_key][-1])
            cis.append(s.ys[f"{y_key}_ci"][-1])
        series.ys[label] = means
        series.ys[f"{label}_ci"] = cis
    return series


def series_to_plot_input(
    series: Series,
) -> list[tuple[str, str, list[float], list[float], list[float]]]:
    '''Drop NaN entries per dataset before passing to plot_shaded_error.'''
    out: list[tuple[str, str, list[float], list[float], list[float]]] = []
    for label, _ in SERIES:
        means_full = series.ys[label]
        cis_full = series.ys[f"{label}_ci"]
        lambdas: list[float] = []
        means: list[float] = []
        cis: list[float] = []
        for lam, m, c in zip(series.x, means_full, cis_full):
            if math.isnan(m):
                continue
            lambdas.append(float(lam))
            means.append(m)
            cis.append(c)
        if not lambdas:
            print(f"  no data for {label}; skipping")
            continue
        out.append((label, "±95% CI", lambdas, means, cis))
    return out


def main() -> None:
    print("Building value-head plot...")
    value_series = build_combined_series("value_accuracy")
    save_series(value_series, f"{OUTPUTS_PARENT_DIR}/lambda_sweep_value_accuracy.pkl")
    plot_shaded_error(
        "Final Iteration Value Head Accuracy over td_lambda",
        series_to_plot_input(value_series),  # type: ignore[arg-type]
        x_label="td_lambda",
        y_label="Accuracy",
        fn="final_iter_value_accuracy_over_lambda",
        y_lim=(0, 1),
    )

    print("Building policy-head plot...")
    policy_series = build_combined_series("policy_accuracy")
    save_series(policy_series, f"{OUTPUTS_PARENT_DIR}/lambda_sweep_policy_accuracy.pkl")
    plot_shaded_error(
        "Final Iteration Policy Head Accuracy over td_lambda",
        series_to_plot_input(policy_series),  # type: ignore[arg-type]
        x_label="td_lambda",
        y_label="Accuracy",
        fn="final_iter_policy_accuracy_over_lambda",
        y_lim=(0, 1),
    )


if __name__ == "__main__":
    main()
