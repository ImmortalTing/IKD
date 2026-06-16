import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_METHODS = (
    "mifgsm",
    "difgsm",
    "tifgsm",
    "nifgsm",
    "sinifgsm",
    "vmifgsm",
    "vnifgsm",
)
DEFAULT_WEIGHTS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
METHOD_GROUPS = (
    ("mifgsm", "difgsm", "tifgsm", "nifgsm"),
    ("sinifgsm", "vmifgsm", "vnifgsm"),
)
METHOD_COLORS = {
    "mifgsm": "#989A9C",
    "difgsm": "#F7D08D",
    "tifgsm": "#BF83A5",
    "nifgsm": "#8684B0",
    "sinifgsm": "#F7D08D",
    "vmifgsm": "#BF83A5",
    "vnifgsm": "#8684B0",
}
METHOD_MARKERS = {
    "mifgsm": "o",
    "difgsm": "s",
    "tifgsm": "^",
    "nifgsm": "D",
    "sinifgsm": "v",
    "vmifgsm": "P",
    "vnifgsm": "X",
}
LOG_FILE_NAME = "log (acc, asr).csv"
FIGURE_SIZE = (3.35, 2.25)
BASE_FONT_SIZE = 7
AXES_LABEL_FONT_SIZE = 7
LEGEND_FONT_SIZE = 6.6
LINE_WIDTH = 1.3
MARKER_SIZE = 3.8
MARKER_EDGE_WIDTH = 0.6
BASELINE_ALPHA = 0.35
METHOD_LEGEND_COLUMNS = 2
METHOD_GROUP_LEGEND_LOCS = {
    METHOD_GROUPS[0]: "upper right",
    METHOD_GROUPS[1]: "lower left",
}
STYLE_NOTE_TEXT = "solid: IKD   dashed: Non-IKD baseline"
STYLE_NOTE_FONT_SIZE = 5.2
STYLE_NOTE_POSITIONS = {
    METHOD_GROUPS[0]: (0.02, 0.03, "left"),
    METHOD_GROUPS[1]: (0.98, 0.18, "right"),
}
FIGURE_BOTTOM_MARGIN = 0.24
TICK_LABEL_ROTATION = 30
IKD_LABEL = "IKD"
NON_IKD_BASELINE_LABEL = "Non-IKD baseline"

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
LOG_ROW_PATTERN = re.compile(
    rf"^(?P<model>.+?)_benign_acc:\s*{FLOAT_PATTERN},\s*"
    rf"(?P=model)_benign_asr:\s*{FLOAT_PATTERN},\s*"
    rf"(?P=model)_adv_acc:\s*{FLOAT_PATTERN},\s*"
    rf"(?P=model)_adv_asr:\s*(?P<adv_asr>{FLOAT_PATTERN}),\s*"
    rf"(?P=model)_fdadv_acc:\s*{FLOAT_PATTERN},\s*"
    rf"(?P=model)_fdadv_asr:\s*(?P<fdadv_asr>{FLOAT_PATTERN})\s*$"
)


@dataclass(frozen=True)
class AsrMetrics:
    adv_asr: float
    fdadv_asr: float


@dataclass(frozen=True)
class BlackboxSummary:
    base_asr: float
    fd_asr: float
    num_targets: int


@dataclass(frozen=True)
class WeightRecord:
    method: str
    weight: float
    blackbox_fd_asr: float
    blackbox_base_asr: float
    source_model: str
    regularization: str
    seed: str
    num_blackbox_targets: int


@dataclass(frozen=True)
class FigureOutputPath:
    group_index: int
    methods: tuple
    pdf_path: Path
    png_path: Path


def format_weight_dir(weight):
    return str(float(weight))


def format_weight_label(weight):
    return f"{float(weight):g}"


def parse_log_file(log_path):
    log_path = Path(log_path)
    parsed = {}

    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = LOG_ROW_PATTERN.match(line)
            if match is None:
                raise ValueError(
                    f"Could not parse {log_path} line {line_number}: {line}"
                )

            parsed[match.group("model")] = AsrMetrics(
                adv_asr=float(match.group("adv_asr")),
                fdadv_asr=float(match.group("fdadv_asr")),
            )

    if not parsed:
        raise ValueError(f"No result rows found in {log_path}")

    return parsed


def summarize_blackbox_asr(metrics_by_model, source_model):
    blackbox_metrics = [
        metrics
        for model_name, metrics in metrics_by_model.items()
        if model_name != source_model
    ]
    if not blackbox_metrics:
        raise ValueError(f"No black-box target rows found for source model {source_model}")

    return BlackboxSummary(
        base_asr=sum(metrics.adv_asr for metrics in blackbox_metrics)
        / len(blackbox_metrics),
        fd_asr=sum(metrics.fdadv_asr for metrics in blackbox_metrics)
        / len(blackbox_metrics),
        num_targets=len(blackbox_metrics),
    )


def result_log_path(results_dir, method, source_model, regularization, seed, weight):
    return (
        Path(results_dir)
        / method
        / source_model
        / regularization
        / str(seed)
        / format_weight_dir(weight)
        / LOG_FILE_NAME
    )


def load_blackbox_summary(
    results_dir, method, source_model, regularization, seed, weight
):
    log_path = result_log_path(
        results_dir=results_dir,
        method=method,
        source_model=source_model,
        regularization=regularization,
        seed=seed,
        weight=weight,
    )
    if not log_path.exists():
        raise FileNotFoundError(
            "Missing result log for "
            f"method={method}, source_model={source_model}, "
            f"regularization={regularization}, seed={seed}, "
            f"weight={format_weight_label(weight)}: {log_path}"
        )

    return summarize_blackbox_asr(parse_log_file(log_path), source_model)


def collect_method_records(
    results_dir,
    method,
    source_model,
    regularization,
    seed,
    weights,
    baseline_weight,
):
    baseline_summary = load_blackbox_summary(
        results_dir=results_dir,
        method=method,
        source_model=source_model,
        regularization=regularization,
        seed=seed,
        weight=baseline_weight,
    )

    records = []
    for weight in weights:
        summary = load_blackbox_summary(
            results_dir=results_dir,
            method=method,
            source_model=source_model,
            regularization=regularization,
            seed=seed,
            weight=weight,
        )
        records.append(
            WeightRecord(
                method=method,
                weight=float(weight),
                blackbox_fd_asr=summary.fd_asr,
                blackbox_base_asr=baseline_summary.base_asr,
                source_model=source_model,
                regularization=regularization,
                seed=str(seed),
                num_blackbox_targets=summary.num_targets,
            )
        )

    return records


def collect_all_records(
    results_dir,
    methods,
    source_model,
    regularization,
    seed,
    weights,
    baseline_weight,
):
    records_by_method = {}
    for method in methods:
        records_by_method[method] = collect_method_records(
            results_dir=results_dir,
            method=method,
            source_model=source_model,
            regularization=regularization,
            seed=seed,
            weights=weights,
            baseline_weight=baseline_weight,
        )
    return records_by_method


def write_summary_csv(records_by_method, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "weight",
        "blackbox_fd_asr",
        "blackbox_base_asr",
        "source_model",
        "regularization",
        "seed",
        "num_blackbox_targets",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in DEFAULT_METHODS:
            for record in records_by_method.get(method, []):
                writer.writerow(
                    {
                        "method": record.method,
                        "weight": format_weight_label(record.weight),
                        "blackbox_fd_asr": f"{record.blackbox_fd_asr:.4f}",
                        "blackbox_base_asr": f"{record.blackbox_base_asr:.4f}",
                        "source_model": record.source_model,
                        "regularization": record.regularization,
                        "seed": record.seed,
                        "num_blackbox_targets": record.num_blackbox_targets,
                    }
                )


def method_display_name(method):
    return method.upper()


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


def compute_y_limits(records_by_method):
    values = []
    for records in records_by_method.values():
        values.extend(record.blackbox_fd_asr for record in records)
        values.extend(record.blackbox_base_asr for record in records)

    if not values:
        return 0.0, 100.0

    lower = max(0.0, min(values) - 3.0)
    upper = min(100.0, max(values) + 3.0)
    if upper - lower < 10.0:
        midpoint = (lower + upper) / 2.0
        lower = max(0.0, midpoint - 5.0)
        upper = min(100.0, midpoint + 5.0)
    return lower, upper


def plot_method_group(
    records_by_method,
    method_group,
    pdf_path,
    png_path,
    y_limits,
    weights=DEFAULT_WEIGHTS,
):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = BASE_FONT_SIZE
    plt.rcParams["axes.labelsize"] = AXES_LABEL_FONT_SIZE
    plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE

    tick_labels = [format_weight_label(weight) for weight in weights]
    y_lower, y_upper = y_limits

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for method in method_group:
        records = records_by_method[method]
        x_values = [record.weight for record in records]
        fd_values = [record.blackbox_fd_asr for record in records]
        baseline = records[0].blackbox_base_asr
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]

        ax.plot(
            x_values,
            fd_values,
            color=color,
            marker=marker,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=method_display_name(method),
        )
        ax.plot(
            x_values,
            [baseline] * len(x_values),
            color=color,
            linestyle="--",
            marker=marker,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            alpha=BASELINE_ALPHA,
        )

    ax.set_xlabel("Weight")
    ax.set_ylabel("Average Black-box ASR (%)")
    ax.set_xscale("log")
    ax.set_xlim(min(weights) * 0.75, max(weights) * 1.35)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(weights)
    ax.set_xticklabels(
        tick_labels,
        rotation=TICK_LABEL_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.3, alpha=0.35)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=method_display_name(method),
        )
        for method in method_group
    ]
    method_legend = ax.legend(
        handles=method_handles,
        loc=METHOD_GROUP_LEGEND_LOCS[tuple(method_group)],
        ncol=min(METHOD_LEGEND_COLUMNS, len(method_group)),
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="#cccccc",
        borderpad=0.25,
        labelspacing=0.25,
        columnspacing=0.7,
        handlelength=1.35,
        handletextpad=0.45,
    )
    method_legend.get_frame().set_linewidth(0.4)

    note_x, note_y, note_ha = STYLE_NOTE_POSITIONS[tuple(method_group)]
    ax.text(
        note_x,
        note_y,
        STYLE_NOTE_TEXT,
        transform=ax.transAxes,
        ha=note_ha,
        va="bottom",
        fontsize=STYLE_NOTE_FONT_SIZE,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "linewidth": 0.35,
            "alpha": 0.72,
        },
    )

    fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=FIGURE_BOTTOM_MARGIN)

    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_weight_analysis(
    records_by_method,
    figure_paths,
    source_model,
    regularization,
    seed,
    weights=DEFAULT_WEIGHTS,
):
    y_limits = compute_y_limits(records_by_method)
    for figure_path in figure_paths:
        plot_method_group(
            records_by_method=records_by_method,
            method_group=figure_path.methods,
            pdf_path=figure_path.pdf_path,
            png_path=figure_path.png_path,
            y_limits=y_limits,
            weights=weights,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot black-box ASR weight analysis for IKD attack variants."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--source-model", default="resnet50")
    parser.add_argument("--regularization", default="KL")
    parser.add_argument("--seed", default="1111")
    parser.add_argument("--baseline-weight", type=float, default=0.01)
    parser.add_argument("--output-dir", default="figures/parameter_analysis")
    parser.add_argument("--output-prefix", default="weight_blackbox_asr")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    records_by_method = collect_all_records(
        results_dir=args.results_dir,
        methods=DEFAULT_METHODS,
        source_model=args.source_model,
        regularization=args.regularization,
        seed=args.seed,
        weights=DEFAULT_WEIGHTS,
        baseline_weight=args.baseline_weight,
    )

    csv_path = output_dir / f"{args.output_prefix}.csv"
    figure_paths = build_figure_output_paths(output_dir, args.output_prefix)

    write_summary_csv(records_by_method, csv_path)
    plot_weight_analysis(
        records_by_method=records_by_method,
        figure_paths=figure_paths,
        source_model=args.source_model,
        regularization=args.regularization,
        seed=args.seed,
    )

    print(f"Wrote {csv_path}")
    for figure_path in figure_paths:
        print(f"Wrote {figure_path.pdf_path}")
        print(f"Wrote {figure_path.png_path}")


if __name__ == "__main__":
    main()
