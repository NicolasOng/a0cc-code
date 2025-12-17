"""Simple CSV -> Heatmap plotting utility.

This module provides `plot_csv_heatmap` which loads a CSV and draws a
heatmap of a value column across two axes (defaults: `dataset size`,
`model size`, `final policy acc`). It intentionally does NOT use a
command-line interface; instead call the function directly from Python.

Example:
  from scripts import model_graph
  model_graph.plot_csv_heatmap('results.csv', output='heat.png', annot=True)
"""

from __future__ import annotations

from typing import Optional

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_csv(path: str) -> pd.DataFrame:
	return pd.read_csv(path)


def _prepare_pivot(df: pd.DataFrame, xcol: str, ycol: str, valcol: str) -> pd.DataFrame:
	if xcol not in df.columns or ycol not in df.columns or valcol not in df.columns:
		raise KeyError(f"CSV missing one of the columns: {xcol}, {ycol}, {valcol}")

	pivot = df.pivot_table(index=ycol, columns=xcol, values=valcol, aggfunc="mean")

	# Try numeric sorting for nicer axis ordering, fall back to string-sort.
	try:
		pivot.columns = pd.to_numeric(pivot.columns)
		pivot = pivot.reindex(sorted(pivot.columns), axis=1)
	except Exception:
		pivot = pivot.reindex(sorted(map(str, pivot.columns)), axis=1)

	try:
		pivot.index = pd.to_numeric(pivot.index)
		pivot = pivot.reindex(sorted(pivot.index), axis=0)
	except Exception:
		pivot = pivot.reindex(sorted(map(str, pivot.index)), axis=0)

	return pivot


def plot_heatmap(
	pivot: pd.DataFrame,
	xlabel: str,
	ylabel: str,
	title: Optional[str] = None,
	output: Optional[str] = None,
	annot: bool = False,
	cmap: str = "viridis",
	log_values: bool = False,
	vmin: Optional[float] = None,
	vmax: Optional[float] = None,
):
	data = pivot.copy()
	if log_values:
		data = np.log10(data.astype(float))

	plt.figure(figsize=(max(6, data.shape[1] * 0.6), max(4, data.shape[0] * 0.5)))
	sns.set(style="white")

	ax = sns.heatmap(
		data,
		annot=annot,
		fmt=".2f" if annot else "",
		cmap=cmap,
		linewidths=0.5,
		linecolor="gray",
		vmin=vmin,
		vmax=vmax,
		cbar_kws={"shrink": 0.8},
	)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title)

	plt.tight_layout()
	if output:
		plt.savefig(output, dpi=200)
		print(f"Saved heatmap to {output}")

	return ax


def plot_csv_heatmap(
	csv_path: str,
	x: str = "dataset size",
	y: str = "model size",
	val: str = "final policy acc",
	output: Optional[str] = "heatmap.png",
	annot: bool = False,
	cmap: str = "viridis",
	log_values: bool = False,
	vmin: Optional[float] = None,
	vmax: Optional[float] = None,
	show: bool = False,
):
	"""Load `csv_path` and plot a heatmap of `val` across `y` (rows) and `x` (columns).

	Returns the matplotlib Axes for further customization.
	"""
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	df = load_csv(csv_path)
	pivot = _prepare_pivot(df, x, y, val)

	if pivot.empty:
		raise ValueError("Pivot table is empty; check input columns and data.")

	title = f"{val} across {y} vs {x}"
	ax = plot_heatmap(
		pivot,
		xlabel=x,
		ylabel=y,
		title=title,
		output=output,
		annot=annot,
		cmap=cmap,
		log_values=log_values,
		vmin=vmin,
		vmax=vmax,
	)

	if show:
		plt.show()

	return ax


__all__ = ["plot_csv_heatmap", "plot_lines_by_dataset"]


def plot_lines_by_dataset(
	csv_path: str,
	model_col: str = "model size",
	dataset_col: str = "dataset size",
	val_col: str = "final policy acc",
	output: Optional[str] = None,
	aggfunc: str = "mean",
	marker: str = "o",
	cmap: str = "tab10",
	show: bool = False,
):
	"""Plot lines of `val_col` (y) vs `model_col` (x), one line per `dataset_col`.

	The function groups by `dataset_col` and, within each group, aggregates
	values by `model_col` using `aggfunc` (default: mean) so repeated model
	sizes are summarized.

	Returns the matplotlib Axes.
	"""
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	df = load_csv(csv_path)
	if model_col not in df.columns or dataset_col not in df.columns or val_col not in df.columns:
		raise KeyError(f"CSV missing one of the columns: {model_col}, {dataset_col}, {val_col}")

	# Coerce model sizes to numeric when possible for sensible sorting
	df = df.copy()
	try:
		df[model_col] = pd.to_numeric(df[model_col])
	except Exception:
		pass

	datasets = sorted(df[dataset_col].unique(), key=lambda v: (float(v) if _is_number(v) else str(v)))

	plt.figure(figsize=(8, 5))
	sns.set(style="whitegrid")
	cmap_obj = plt.get_cmap(cmap)

	ax = None
	for i, ds in enumerate(datasets):
		sub = df[df[dataset_col] == ds]
		if sub.empty:
			continue

		# aggregate by model_col
		try:
			grouped = sub.groupby(model_col)[val_col].agg(aggfunc).reset_index()
		except Exception:
			grouped = sub.groupby(model_col)[val_col].mean().reset_index()

		# drop NaNs and sort by model_col
		grouped = grouped.dropna()
		grouped = grouped.sort_values(by=grouped.columns[0])

		if grouped.empty:
			continue

		x = grouped.iloc[:, 0].values
		y = grouped.iloc[:, 1].values

		color = cmap_obj(i % cmap_obj.N)
		ax = plt.plot(x, y, marker=marker, label=str(ds), color=color)

	plt.xlabel(model_col)
	plt.ylabel(val_col)
	plt.title(f"{val_col} vs {model_col} (lines per {dataset_col})")
	plt.legend(title=dataset_col)
	plt.tight_layout()

	if output:
		plt.savefig(output, dpi=200)
		print(f"Saved line plot to {output}")

	if show:
		plt.show()

	# Return current Axes object
	return plt.gca()


def _is_number(v) -> bool:
	try:
		float(v)
		return True
	except Exception:
		return False

plot_lines_by_dataset("results.csv", show=True)