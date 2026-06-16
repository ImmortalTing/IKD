import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_weight_analysis import (
    DEFAULT_METHODS,
    FIGURE_SIZE,
    METHOD_GROUPS,
    format_weight_label,
    load_blackbox_summary,
    method_display_name,
)


IKD_METHODS = ("w/o IKD", "MSE", "CE", "KL")
IKD_METHOD_COLORS = {
    "w/o IKD": "#989A9C",
    "MSE": "#F7D08D",
    "CE": "#BF83A5",
    "KL": "#8684B0",
}
IKD_METHOD_DISPLAY_LABELS = {
    "w/o IKD": "w/o IKD",
    "MSE": "MSE",
    "CE": "CE",
    "KL": "KL",
}
DEFAULT_SOURCE_MODEL = "resnet50"
DEFAULT_SEED = "1111"
DEFAULT_WEIGHT = 0.01
DEFAULT_OUTPUT_PREFIX = "ikd_ablation_blackbox_asr"
BASE_FONT_SIZE = 7
AXES_LABEL_FONT_SIZE = 7
LEGEND_FONT_SIZE = 6.2
BAR_WIDTH = 0.18
BAR_EDGE_COLOR = "#555555"
BAR_EDGE_WIDTH = 0.25
FIGURE_BOTTOM_MARGIN = 0.19
FIGURE_TOP_MARGIN = 0.84


@dataclass(frozen=True)
class IkdAblationRecord:
    method: str
    ikd_method: str
    blackbox_asr: float
    source_model: str
    seed: str
    weight: float
    num_blackbox_targets: int


@dataclass(frozen=True)
class FigureOutputPath:
    group_index: int
    methods: tuple
    pdf_path: Path
    png_path: Path


def build_figure_output_paths(output_dir, output_prefix):
    output_dir = Path(output_dir)
    return tuple(
        FigureOutputPath(
            group_index=index,
            methods=methods,
            pdf_path=output_dir / f"{output_prefix}_group{index}.pdf",
            png_path=output_dir / f"{output_prefix}_group{index}.png",
        )
        for index, methods in enumerate(METHOD_GROUPS, start=1)
    )


def load_ikd_ablation_record(
    results_dir,
    method,
    ikd_method,
    source_model,
    seed,
    weight,
):
    regularization = "KL" if ikd_method == "w/o IKD" else ikd_method
    summary = load_blackbox_summary(
        results_dir=results_dir,
        method=method,
        source_model=source_model,
        regularization=regularization,
        seed=seed,
        weight=weight,
    )
    blackbox_asr = summary.base_asr if ikd_method == "w/o IKD" else summary.ikd_asr

    return IkdAblationRecord(
        method=method,
        ikd_method=ikd_method,
        blackbox_asr=blackbox_asr,
        source_model=source_model,
        seed=str(seed),
        weight=float(weight),
        num_blackbox_targets=summary.num_targets,
    )


def collect_method_records(
    results_dir,
    method,
    source_model=DEFAULT_SOURCE_MODEL,
    seed=DEFAULT_SEED,
    weight=DEFAULT_WEIGHT,
):
    return [
        load_ikd_ablation_record(
            results_dir=results_dir,
            method=method,
            ikd_method=ikd_method,
            source_model=source_model,
            seed=seed,
            weight=weight,
        )
        for ikd_method in IKD_METHODS
    ]


def collect_all_records(
    results_dir,
    methods=DEFAULT_METHODS,
    source_model=DEFAULT_SOURCE_MODEL,
    seed=DEFAULT_SEED,
    weight=DEFAULT_WEIGHT,
):
    return {
        method: collect_method_records(
            results_dir=results_dir,
            method=method,
            source_model=source_model,
            seed=seed,
            weight=weight,
        )
        for method in methods
    }


def write_summary_csv(records_by_method, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "ikd_method",
        "blackbox_asr",
        "source_model",
        "seed",
        "weight",
        "num_blackbox_targets",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in DEFAULT_METHODS:
            records_by_ikd_method = {
                record.ikd_method: record for record in records_by_method.get(method, [])
            }
            for ikd_method in IKD_METHODS:
                record = records_by_ikd_method.get(ikd_method)
                if record is None:
                    continue
                writer.writerow(
                    {
                        "method": record.method,
                        "ikd_method": record.ikd_method,
                        "blackbox_asr": f"{record.blackbox_asr:.4f}",
                        "source_model": record.source_model,
                        "seed": record.seed,
                        "weight": format_weight_label(record.weight),
                        "num_blackbox_targets": record.num_blackbox_targets,
                    }
                )


def compute_y_limits(records_by_method):
    values = [
        record.blackbox_asr
        for records in records_by_method.values()
        for record in records
    ]
    if not values:
        return 0.0, 100.0

    upper = math.ceil(max(values) / 5.0) * 5.0
    upper = min(100.0, max(5.0, upper))
    return 0.0, upper


def records_by_ikd_method(records):
    return {record.ikd_method: record for record in records}


def plot_method_group(records_by_method, method_group, pdf_path, png_path, y_limits):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = BASE_FONT_SIZE
    plt.rcParams["axes.labelsize"] = AXES_LABEL_FONT_SIZE
    plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x_positions = list(range(len(method_group)))
    offsets = [
        (index - (len(IKD_METHODS) - 1) / 2.0) * BAR_WIDTH
        for index in range(len(IKD_METHODS))
    ]

    for ikd_index, ikd_method in enumerate(IKD_METHODS):
        values = []
        for method in method_group:
            method_records = records_by_ikd_method(records_by_method[method])
            values.append(method_records[ikd_method].blackbox_asr)

        ax.bar(
            [x + offsets[ikd_index] for x in x_positions],
            values,
            width=BAR_WIDTH,
            color=IKD_METHOD_COLORS[ikd_method],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label=IKD_METHOD_DISPLAY_LABELS[ikd_method],
        )

    ax.set_xlabel("Attack Method")
    ax.set_ylabel("Average Black-box ASR (%)")
    ax.set_ylim(*y_limits)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([method_display_name(method) for method in method_group])
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=len(IKD_METHODS),
        frameon=True,
        framealpha=0.82,
        facecolor="white",
        edgecolor="#cccccc",
        borderpad=0.22,
        labelspacing=0.25,
        columnspacing=0.65,
        handlelength=1.05,
        handletextpad=0.35,
    )
    legend.get_frame().set_linewidth(0.4)

    fig.subplots_adjust(
        left=0.17,
        right=0.99,
        top=FIGURE_TOP_MARGIN,
        bottom=FIGURE_BOTTOM_MARGIN,
    )

    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ikd_ablation(records_by_method, figure_paths):
    y_limits = compute_y_limits(records_by_method)
    for figure_path in figure_paths:
        plot_method_group(
            records_by_method=records_by_method,
            method_group=figure_path.methods,
            pdf_path=figure_path.pdf_path,
            png_path=figure_path.png_path,
            y_limits=y_limits,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot IKD ablation black-box ASR for attack variants."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    parser.add_argument("--output-dir", default="figures/parameter_analysis")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    records_by_method = collect_all_records(
        results_dir=args.results_dir,
        methods=DEFAULT_METHODS,
        source_model=args.source_model,
        seed=args.seed,
        weight=args.weight,
    )

    csv_path = output_dir / f"{args.output_prefix}.csv"
    figure_paths = build_figure_output_paths(output_dir, args.output_prefix)

    write_summary_csv(records_by_method, csv_path)
    plot_ikd_ablation(records_by_method, figure_paths)

    print(f"Wrote {csv_path}")
    for figure_path in figure_paths:
        print(f"Wrote {figure_path.pdf_path}")
        print(f"Wrote {figure_path.png_path}")


if __name__ == "__main__":
    main()
