import sys
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
    # usage: python scripts/inspect_series_objects.py <series.pkl>
    # e.g. an eval/merged_*_gtv_eval.pkl from a combined sweep
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/inspect_series_objects.py <series.pkl>")
    series = load_series(sys.argv[1])
    #inspect_series_object(series)
    inspect_series_object_given_keys(series, ["policy_accuracy", "policy_accuracy_ci"])