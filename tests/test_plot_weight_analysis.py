import tempfile
import unittest
from pathlib import Path

from plot_weight_analysis import (
    AsrMetrics,
    DEFAULT_METHODS,
    FIGURE_SIZE,
    FIGURE_BOTTOM_MARGIN,
    IKD_LABEL,
    METHOD_GROUPS,
    METHOD_COLORS,
    METHOD_GROUP_LEGEND_LOCS,
    METHOD_LEGEND_COLUMNS,
    METHOD_MARKERS,
    NON_IKD_BASELINE_LABEL,
    STYLE_NOTE_FONT_SIZE,
    STYLE_NOTE_POSITIONS,
    STYLE_NOTE_TEXT,
    TICK_LABEL_ROTATION,
    build_figure_output_paths,
    collect_method_records,
    parse_log_file,
    summarize_blackbox_asr,
)


LOG_NAME = "log (acc, asr).csv"


def write_log(root, method, source_model, regularization, seed, weight, rows):
    log_dir = (
        Path(root)
        / method
        / source_model
        / regularization
        / seed
        / str(weight)
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_NAME
    log_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return log_path


def result_row(model, adv_asr, fdadv_asr):
    return (
        f"{model}_benign_acc: 90.0000, {model}_benign_asr: 10.0000, "
        f"{model}_adv_acc: {100 - adv_asr:.4f}, {model}_adv_asr: {adv_asr:.4f}, "
        f"{model}_fdadv_acc: {100 - fdadv_asr:.4f}, "
        f"{model}_fdadv_asr: {fdadv_asr:.4f}"
    )


class PlotWeightAnalysisTests(unittest.TestCase):
    def test_visible_style_legend_labels_use_ikd_terms(self):
        self.assertEqual(IKD_LABEL, "IKD")
        self.assertEqual(NON_IKD_BASELINE_LABEL, "Non-IKD baseline")

    def test_method_groups_match_requested_split(self):
        expected_groups = (
            ("mifgsm", "difgsm", "tifgsm", "nifgsm"),
            ("sinifgsm", "vmifgsm", "vnifgsm"),
        )

        self.assertEqual(METHOD_GROUPS, expected_groups)
        flattened = [method for group in METHOD_GROUPS for method in group]
        self.assertEqual(flattened, list(DEFAULT_METHODS))

    def test_build_figure_output_paths_uses_group_suffixes(self):
        paths = build_figure_output_paths(
            output_dir=Path("figures/parameter_analysis"),
            output_prefix="weight_blackbox_asr",
        )

        self.assertEqual(
            [(path.pdf_path.name, path.png_path.name) for path in paths],
            [
                ("weight_blackbox_asr_group1.pdf", "weight_blackbox_asr_group1.png"),
                ("weight_blackbox_asr_group2.pdf", "weight_blackbox_asr_group2.png"),
            ],
        )

    def test_method_markers_are_fixed_and_unique(self):
        expected_markers = {
            "mifgsm": "o",
            "difgsm": "s",
            "tifgsm": "^",
            "nifgsm": "D",
            "sinifgsm": "v",
            "vmifgsm": "P",
            "vnifgsm": "X",
        }

        self.assertEqual(METHOD_MARKERS, expected_markers)
        self.assertEqual(set(METHOD_MARKERS), set(DEFAULT_METHODS))
        self.assertEqual(len(set(METHOD_MARKERS.values())), len(DEFAULT_METHODS))

    def test_method_colors_use_requested_palette(self):
        expected_colors = {
            "mifgsm": "#989A9C",
            "difgsm": "#F7D08D",
            "tifgsm": "#BF83A5",
            "nifgsm": "#8684B0",
            "sinifgsm": "#F7D08D",
            "vmifgsm": "#BF83A5",
            "vnifgsm": "#8684B0",
        }

        self.assertEqual(METHOD_COLORS, expected_colors)

    def test_figure_layout_is_compact_for_two_column_latex(self):
        self.assertEqual(FIGURE_SIZE, (3.35, 2.25))
        self.assertLessEqual(FIGURE_SIZE[0], 3.5)
        self.assertLessEqual(FIGURE_BOTTOM_MARGIN, 0.28)
        self.assertEqual(METHOD_LEGEND_COLUMNS, 2)
        self.assertEqual(TICK_LABEL_ROTATION, 30)
        self.assertEqual(
            METHOD_GROUP_LEGEND_LOCS,
            {
                ("mifgsm", "difgsm", "tifgsm", "nifgsm"): "upper right",
                ("sinifgsm", "vmifgsm", "vnifgsm"): "lower left",
            },
        )
        self.assertEqual(STYLE_NOTE_TEXT, "solid: IKD   dashed: Non-IKD baseline")
        self.assertLessEqual(STYLE_NOTE_FONT_SIZE, 5.4)
        self.assertEqual(STYLE_NOTE_POSITIONS[METHOD_GROUPS[0]], (0.02, 0.03, "left"))
        self.assertEqual(STYLE_NOTE_POSITIONS[METHOD_GROUPS[1]], (0.98, 0.18, "right"))

    def test_parse_log_file_handles_model_names_with_underscores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / LOG_NAME
            log_path.write_text(
                "\n".join(
                    [
                        result_row("resnet50", adv_asr=90.0, fdadv_asr=91.0),
                        result_row(
                            "ens_adv_inception_resnet_v2",
                            adv_asr=12.5,
                            fdadv_asr=34.5,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = parse_log_file(log_path)

        self.assertIn("ens_adv_inception_resnet_v2", parsed)
        self.assertEqual(parsed["ens_adv_inception_resnet_v2"].adv_asr, 12.5)
        self.assertEqual(parsed["ens_adv_inception_resnet_v2"].fdadv_asr, 34.5)

    def test_summarize_blackbox_asr_excludes_source_model(self):
        parsed = {
            "resnet50": AsrMetrics(adv_asr=99.0, fdadv_asr=98.0),
            "densenet121": AsrMetrics(adv_asr=10.0, fdadv_asr=20.0),
            "inc_v3": AsrMetrics(adv_asr=30.0, fdadv_asr=40.0),
        }

        summary = summarize_blackbox_asr(parsed, source_model="resnet50")

        self.assertEqual(summary.num_targets, 2)
        self.assertEqual(summary.base_asr, 20.0)
        self.assertEqual(summary.ikd_asr, 30.0)

    def test_collect_method_records_raises_for_missing_weight_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                result_row("resnet50", adv_asr=99.0, fdadv_asr=98.0),
                result_row("densenet121", adv_asr=10.0, fdadv_asr=20.0),
            ]
            write_log(tmpdir, "mifgsm", "resnet50", "KL", "1111", "0.01", rows)

            with self.assertRaisesRegex(
                FileNotFoundError,
                r"Missing result log.*mifgsm.*0\.001",
            ):
                collect_method_records(
                    results_dir=tmpdir,
                    method="mifgsm",
                    source_model="resnet50",
                    regularization="KL",
                    seed="1111",
                    weights=[0.001, 0.01],
                    baseline_weight=0.01,
                )

    def test_collect_method_records_uses_requested_baseline_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_log(
                tmpdir,
                "mifgsm",
                "resnet50",
                "KL",
                "1111",
                "0.01",
                [
                    result_row("resnet50", adv_asr=99.0, fdadv_asr=98.0),
                    result_row("densenet121", adv_asr=10.0, fdadv_asr=20.0),
                    result_row("inc_v3", adv_asr=20.0, fdadv_asr=30.0),
                ],
            )
            write_log(
                tmpdir,
                "mifgsm",
                "resnet50",
                "KL",
                "1111",
                "1.0",
                [
                    result_row("resnet50", adv_asr=99.0, fdadv_asr=98.0),
                    result_row("densenet121", adv_asr=50.0, fdadv_asr=60.0),
                    result_row("inc_v3", adv_asr=70.0, fdadv_asr=80.0),
                ],
            )

            records = collect_method_records(
                results_dir=tmpdir,
                method="mifgsm",
                source_model="resnet50",
                regularization="KL",
                seed="1111",
                weights=[0.01, 1.0],
                baseline_weight=1.0,
        )

        self.assertEqual([record.weight for record in records], [0.01, 1.0])
        self.assertEqual([record.blackbox_ikd_asr for record in records], [25.0, 70.0])
        self.assertEqual([record.blackbox_base_asr for record in records], [60.0, 60.0])


if __name__ == "__main__":
    unittest.main()
