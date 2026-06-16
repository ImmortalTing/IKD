import csv
import tempfile
import unittest
from pathlib import Path

from plot_fd_ablation import (
    DEFAULT_METHODS,
    FD_METHODS,
    FD_METHOD_COLORS,
    FD_METHOD_DISPLAY_LABELS,
    FIGURE_SIZE,
    METHOD_GROUPS,
    FdAblationRecord,
    build_figure_output_paths,
    collect_method_records,
    compute_y_limits,
    write_summary_csv,
)


LOG_NAME = "log (acc, asr).csv"


def result_row(model, adv_asr, fdadv_asr):
    return (
        f"{model}_benign_acc: 90.0000, {model}_benign_asr: 10.0000, "
        f"{model}_adv_acc: {100 - adv_asr:.4f}, {model}_adv_asr: {adv_asr:.4f}, "
        f"{model}_fdadv_acc: {100 - fdadv_asr:.4f}, "
        f"{model}_fdadv_asr: {fdadv_asr:.4f}"
    )


def write_log(root, method, source_model, regularization, seed, weight, rows):
    log_dir = (
        Path(root)
        / method
        / source_model
        / regularization
        / seed
        / str(float(weight))
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_NAME
    log_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return log_path


class PlotFdAblationTests(unittest.TestCase):
    def test_method_groups_and_fd_method_order_match_requested_ablation(self):
        self.assertEqual(
            METHOD_GROUPS,
            (
                ("mifgsm", "difgsm", "tifgsm", "nifgsm"),
                ("sinifgsm", "vmifgsm", "vnifgsm"),
            ),
        )
        self.assertEqual(
            DEFAULT_METHODS,
            (
                "mifgsm",
                "difgsm",
                "tifgsm",
                "nifgsm",
                "sinifgsm",
                "vmifgsm",
                "vnifgsm",
            ),
        )
        self.assertEqual(FD_METHODS, ("w/o FD", "MSE", "CE", "KL"))

    def test_fd_method_colors_use_requested_palette(self):
        self.assertEqual(
            FD_METHOD_COLORS,
            {
                "w/o FD": "#989A9C",
                "MSE": "#F7D08D",
                "CE": "#BF83A5",
                "KL": "#8684B0",
            },
        )
        self.assertEqual(FIGURE_SIZE, (3.35, 2.25))

    def test_fd_method_display_labels_use_ikd_wording(self):
        self.assertEqual(
            tuple(FD_METHOD_DISPLAY_LABELS[fd_method] for fd_method in FD_METHODS),
            ("w/o IKD", "MSE", "CE", "KL"),
        )

    def test_build_figure_output_paths_uses_fd_ablation_group_suffixes(self):
        paths = build_figure_output_paths(
            output_dir=Path("figures/parameter_analysis"),
            output_prefix="fd_ablation_blackbox_asr",
        )

        self.assertEqual(
            [(path.pdf_path.name, path.png_path.name) for path in paths],
            [
                (
                    "fd_ablation_blackbox_asr_group1.pdf",
                    "fd_ablation_blackbox_asr_group1.png",
                ),
                (
                    "fd_ablation_blackbox_asr_group2.pdf",
                    "fd_ablation_blackbox_asr_group2.png",
                ),
            ],
        )

    def test_collect_method_records_uses_base_asr_for_without_fd_and_fd_asr_for_regularizers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_log(
                tmpdir,
                "mifgsm",
                "resnet50",
                "KL",
                "1111",
                0.01,
                [
                    result_row("resnet50", adv_asr=99.0, fdadv_asr=98.0),
                    result_row("densenet121", adv_asr=10.0, fdadv_asr=80.0),
                    result_row("inc_v3", adv_asr=30.0, fdadv_asr=100.0),
                ],
            )
            write_log(
                tmpdir,
                "mifgsm",
                "resnet50",
                "MSE",
                "1111",
                0.01,
                [
                    result_row("resnet50", adv_asr=99.0, fdadv_asr=98.0),
                    result_row("densenet121", adv_asr=10.0, fdadv_asr=40.0),
                    result_row("inc_v3", adv_asr=30.0, fdadv_asr=60.0),
                ],
            )
            write_log(
                tmpdir,
                "mifgsm",
                "resnet50",
                "CE",
                "1111",
                0.01,
                [
                    result_row("resnet50", adv_asr=99.0, fdadv_asr=98.0),
                    result_row("densenet121", adv_asr=10.0, fdadv_asr=70.0),
                    result_row("inc_v3", adv_asr=30.0, fdadv_asr=90.0),
                ],
            )

            records = collect_method_records(
                results_dir=tmpdir,
                method="mifgsm",
                source_model="resnet50",
                seed="1111",
                weight=0.01,
            )

        self.assertEqual([record.fd_method for record in records], list(FD_METHODS))
        self.assertEqual(
            [record.blackbox_asr for record in records],
            [20.0, 50.0, 80.0, 90.0],
        )
        self.assertEqual([record.num_blackbox_targets for record in records], [2] * 4)

    def test_collect_method_records_raises_for_missing_regularization_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                result_row("resnet50", adv_asr=99.0, fdadv_asr=98.0),
                result_row("densenet121", adv_asr=10.0, fdadv_asr=20.0),
            ]
            write_log(tmpdir, "mifgsm", "resnet50", "KL", "1111", 0.01, rows)
            write_log(tmpdir, "mifgsm", "resnet50", "MSE", "1111", 0.01, rows)

            with self.assertRaisesRegex(
                FileNotFoundError,
                r"Missing result log.*method=mifgsm.*regularization=CE.*weight=0\.01",
            ):
                collect_method_records(
                    results_dir=tmpdir,
                    method="mifgsm",
                    source_model="resnet50",
                    seed="1111",
                    weight=0.01,
                )

    def test_write_summary_csv_uses_requested_fields_and_method_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            records_by_method = {
                "mifgsm": [
                    FdAblationRecord(
                        method="mifgsm",
                        fd_method="w/o FD",
                        blackbox_asr=47.54,
                        source_model="resnet50",
                        seed="1111",
                        weight=0.01,
                        num_blackbox_targets=15,
                    ),
                    FdAblationRecord(
                        method="mifgsm",
                        fd_method="MSE",
                        blackbox_asr=21.9467,
                        source_model="resnet50",
                        seed="1111",
                        weight=0.01,
                        num_blackbox_targets=15,
                    ),
                ],
            }

            write_summary_csv(records_by_method, csv_path)

            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(
            rows,
            [
                {
                    "method": "mifgsm",
                    "fd_method": "w/o FD",
                    "blackbox_asr": "47.5400",
                    "source_model": "resnet50",
                    "seed": "1111",
                    "weight": "0.01",
                    "num_blackbox_targets": "15",
                },
                {
                    "method": "mifgsm",
                    "fd_method": "MSE",
                    "blackbox_asr": "21.9467",
                    "source_model": "resnet50",
                    "seed": "1111",
                    "weight": "0.01",
                    "num_blackbox_targets": "15",
                },
            ],
        )

    def test_compute_y_limits_starts_at_zero_and_rounds_maximum_up(self):
        records_by_method = {
            "mifgsm": [
                FdAblationRecord("mifgsm", "w/o FD", 47.54, "resnet50", "1111", 0.01, 15),
                FdAblationRecord("mifgsm", "KL", 69.01, "resnet50", "1111", 0.01, 15),
            ]
        }

        self.assertEqual(compute_y_limits(records_by_method), (0.0, 70.0))


if __name__ == "__main__":
    unittest.main()
