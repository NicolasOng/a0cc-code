from a0.eval.plotting import Series, load_series

def inspect_series_object(series: Series) -> None:
    for y_key in series.ys.keys():
        print(f"{y_key}, ", end="")
    print()
    for x in series.x:
        for y_key in series.ys.keys():
            y_values = series.ys[y_key]
            print(f"{y_values[x]:.2%}, ", end="")
        print()

def inspect_series_object_given_keys(series: Series, keys: list[str]) -> None:
    for y_key in keys:
        print(f"{y_key}, ", end="")
    print()
    for x in series.x:
        for y_key in keys:
            y_values = series.ys[y_key]
            print(f"{y_values[x]:.2%}, ", end="")
        print()
if __name__ == "__main__":
    series = load_series("/home/nicolas/Downloads/2026-01-12 output-byrrp eval fixed/eval/merged_random_nt_gtv_eval.pkl")
    #series = load_series("/home/nicolas/Downloads/2026-01-12 output-byrrp eval fixed/eval/merged_training_nt_gtv_eval.pkl")
    #inspect_series_object(series)
    inspect_series_object_given_keys(series, ["policy_accuracy", "policy_accuracy_ci"])