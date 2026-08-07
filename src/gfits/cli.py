"""Command-line entry point for completed G-FITS phases."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gfits import __version__
from gfits.e01 import download_vision_e01, run_e01
from gfits.e02 import (
    build_e02_fingerprint_bank,
    evaluate_e02,
    extract_e02_residuals,
    generate_e02_report,
    score_e02_pairs,
)
from gfits.e02_data import prepare_e02_data
from gfits.e02b import (
    apply_e02b_counterfactuals,
    evaluate_e02b,
    extract_e02b_representations,
    generate_e02b_report,
    merge_e02b_representation_fragments,
    score_e02b_matrix,
    select_e02b_conditions,
)
from gfits.e02b_generation import generate_e02b_shard, merge_e02b_fragments
from gfits.manifest import ManifestError, build_manifest, verify_manifest
from gfits.prnu_validation import validate_prnu
from gfits.synthetic import validate_synthetic_fits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gfits",
        description="G-FITS reproducible research utilities (Phase 0 through E02b)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-manifest", help="hash raw dataset files into JSON")
    build.add_argument("--root", type=Path, required=True, help="dataset root")
    build.add_argument("--output", type=Path, required=True, help="manifest JSON path")
    selection = build.add_mutually_exclusive_group()
    selection.add_argument(
        "--include",
        action="append",
        help="include glob; repeat to override default image globs",
    )
    selection.add_argument(
        "--all-files",
        action="store_true",
        help="include every regular file under the root",
    )

    verify = commands.add_parser("verify-manifest", help="verify presence, size, and SHA-256")
    verify.add_argument("--manifest", type=Path, required=True, help="manifest JSON path")
    verify.add_argument("--root", type=Path, help="override the recorded dataset root")
    verify.add_argument(
        "--strict",
        action="store_true",
        help="also fail on included files absent from the manifest",
    )

    prnu = commands.add_parser(
        "validate-prnu",
        help="cross-validate CC and legacy signed PCE against locked prnu-python",
    )
    prnu.add_argument("--upstream-root", type=Path, required=True)
    prnu.add_argument("--output", type=Path, required=True)

    synthetic = commands.add_parser(
        "validate-synthetic-fits",
        help="run the pre-registered E00 residual-level synthetic gate",
    )
    synthetic.add_argument("--config", type=Path, required=True)
    synthetic.add_argument("--output-dir", type=Path, required=True)
    synthetic.add_argument(
        "--profile",
        choices=("development", "gate"),
        default="gate",
    )

    download = commands.add_parser(
        "download-vision-e01",
        help="download and hash the pre-registered VISION E01 subset without transforms",
    )
    download.add_argument("--config", type=Path, required=True)
    download.add_argument("--data-root", type=Path, required=True)
    download.add_argument("--manifest", type=Path, required=True)

    e01 = commands.add_parser(
        "run-e01",
        help="run the pre-registered real downstream-pipeline mechanism replication",
    )
    e01.add_argument("--config", type=Path, required=True)
    e01.add_argument("--manifest", type=Path, required=True)
    e01.add_argument("--data-root", type=Path, required=True)
    e01.add_argument("--upstream-root", type=Path, required=True)
    e01.add_argument("--output-dir", type=Path, required=True)
    e01.add_argument("--cache-root", type=Path, required=True)

    prepare_e02 = commands.add_parser(
        "prepare-e02-data",
        help="range-extract and hash the pre-registered E02 archive members",
    )
    prepare_e02.add_argument("--config", type=Path, required=True)
    prepare_e02.add_argument("--data-root", type=Path, required=True)
    prepare_e02.add_argument("--manifest", type=Path, required=True)

    residuals = commands.add_parser(
        "extract-residuals",
        help="extract one or more registered E02 residual representations",
    )
    residuals.add_argument("--config", type=Path, required=True)
    residuals.add_argument("--manifest", type=Path, required=True)
    residuals.add_argument("--data-root", type=Path, required=True)
    residuals.add_argument("--upstream-root", type=Path, required=True)
    residuals.add_argument("--cache-root", type=Path, required=True)
    residuals.add_argument("--residual-manifest", type=Path, required=True)
    residuals.add_argument(
        "--extractor",
        action="append",
        choices=("wavelet", "srm", "low_bit", "noiseprint"),
        help="repeat to run a subset; the default runs all registered extractors",
    )
    residuals.add_argument("--noiseprint-root", type=Path)

    bank = commands.add_parser(
        "build-fingerprint-bank",
        help="build the complete E02 extractor-by-aggregator fingerprint bank",
    )
    bank.add_argument("--config", type=Path, required=True)
    bank.add_argument("--manifest", type=Path, required=True)
    bank.add_argument("--residual-manifest", type=Path, required=True)
    bank.add_argument("--bank-root", type=Path, required=True)
    bank.add_argument("--bank-manifest", type=Path, required=True)

    score = commands.add_parser(
        "score-pairs",
        help="score E02 calibration/test queries against native-geometry fingerprints",
    )
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--residual-manifest", type=Path, required=True)
    score.add_argument("--bank-manifest", type=Path, required=True)
    score.add_argument("--scores", type=Path, required=True)

    evaluate = commands.add_parser(
        "evaluate-attribution",
        help="run the frozen E02 paired tests and mainline stop/go gate",
    )
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--residual-manifest", type=Path, required=True)
    evaluate.add_argument("--scores", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)

    report = commands.add_parser(
        "generate-report",
        help="render the E02 report from frozen machine-readable evidence",
    )
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--summary", type=Path, required=True)
    report.add_argument("--evaluation", type=Path, required=True)
    report.add_argument("--report", type=Path, required=True)

    generate_e02b = commands.add_parser(
        "generate-e02b-shard",
        help="generate one committed, revision-locked E02b model/resolution shard",
    )
    generate_e02b.add_argument("--config", type=Path, required=True)
    generate_e02b.add_argument("--repository-root", type=Path, required=True)
    generate_e02b.add_argument("--data-root", type=Path, required=True)
    generate_e02b.add_argument("--cache-root", type=Path, required=True)
    generate_e02b.add_argument("--fragment", type=Path, required=True)
    generate_e02b.add_argument("--model-id", required=True)
    generate_e02b.add_argument("--resolution", type=int, required=True)
    generate_e02b.add_argument("--device", default="cuda:0")
    generate_e02b.add_argument("--batch-size", type=int, default=1)

    merge_generation = commands.add_parser(
        "merge-e02b-generation",
        help="merge and verify all E02b generation fragments",
    )
    merge_generation.add_argument("--config", type=Path, required=True)
    merge_generation.add_argument("--repository-root", type=Path, required=True)
    merge_generation.add_argument("--data-root", type=Path, required=True)
    merge_generation.add_argument("--fragment-root", type=Path, required=True)
    merge_generation.add_argument("--output", type=Path, required=True)

    counterfactuals = commands.add_parser(
        "apply-e02b-counterfactuals",
        help="create the registered E02b exporter and low-bit counterfactuals",
    )
    counterfactuals.add_argument("--config", type=Path, required=True)
    counterfactuals.add_argument("--repository-root", type=Path, required=True)
    counterfactuals.add_argument("--generation-manifest", type=Path, required=True)
    counterfactuals.add_argument("--data-root", type=Path, required=True)
    counterfactuals.add_argument("--derivative-root", type=Path, required=True)
    counterfactuals.add_argument("--output", type=Path, required=True)

    extract_e02b = commands.add_parser(
        "extract-e02b-representations",
        help="extract one E02b residual plus its registered statistical signatures",
    )
    extract_e02b.add_argument("--config", type=Path, required=True)
    extract_e02b.add_argument("--repository-root", type=Path, required=True)
    extract_e02b.add_argument("--generation-manifest", type=Path, required=True)
    extract_e02b.add_argument("--data-root", type=Path, required=True)
    extract_e02b.add_argument("--cache-root", type=Path, required=True)
    extract_e02b.add_argument("--fragment", type=Path, required=True)
    extract_e02b.add_argument("--upstream-root", type=Path, required=True)
    extract_e02b.add_argument(
        "--extractor",
        required=True,
        choices=(
            "wavelet",
            "fixed_three_kernel_high_pass_residual_bank",
            "low_bit",
            "noiseprint",
        ),
    )
    extract_e02b.add_argument("--condition", default="native")
    extract_e02b.add_argument("--derivative-manifest", type=Path)
    extract_e02b.add_argument("--derivative-root", type=Path)
    extract_e02b.add_argument("--noiseprint-root", type=Path)

    merge_representations = commands.add_parser(
        "merge-e02b-representations",
        help="merge and hash-verify E02b representation fragments",
    )
    merge_representations.add_argument("--config", type=Path, required=True)
    merge_representations.add_argument("--repository-root", type=Path, required=True)
    merge_representations.add_argument("--fragment-root", type=Path, required=True)
    merge_representations.add_argument("--cache-root", type=Path, required=True)
    merge_representations.add_argument("--output", type=Path, required=True)

    score_e02b = commands.add_parser(
        "score-e02b",
        help="score the E02b calibration/test matrix with explicit scorer semantics",
    )
    score_e02b.add_argument("--config", type=Path, required=True)
    score_e02b.add_argument("--repository-root", type=Path, required=True)
    score_e02b.add_argument("--generation-manifest", type=Path, required=True)
    score_e02b.add_argument("--data-root", type=Path, required=True)
    score_e02b.add_argument("--representation-manifest", type=Path, required=True)
    score_e02b.add_argument("--cache-root", type=Path, required=True)
    score_e02b.add_argument("--scores", type=Path, required=True)
    score_e02b.add_argument("--common-scores", type=Path, required=True)
    score_e02b.add_argument("--calibration", type=Path, required=True)
    score_e02b.add_argument("--condition", default="native")
    score_e02b.add_argument("--derivative-manifest", type=Path)
    score_e02b.add_argument("--derivative-root", type=Path)
    score_e02b.add_argument(
        "--profile",
        choices=("matrix", "counterfactual", "resolution"),
        default="matrix",
    )

    select_e02b = commands.add_parser(
        "select-e02b-condition",
        help="select E02b conditions from calibration rows without reading test scores",
    )
    select_e02b.add_argument("--config", type=Path, required=True)
    select_e02b.add_argument("--scores", type=Path, required=True)
    select_e02b.add_argument("--output", type=Path, required=True)

    evaluate_e02b_parser = commands.add_parser(
        "evaluate-e02b",
        help="run E02b test inference, two-way bootstrap, permutations, and Gate",
    )
    evaluate_e02b_parser.add_argument("--config", type=Path, required=True)
    evaluate_e02b_parser.add_argument("--repository-root", type=Path, required=True)
    evaluate_e02b_parser.add_argument("--generation-manifest", type=Path, required=True)
    evaluate_e02b_parser.add_argument("--data-root", type=Path, required=True)
    evaluate_e02b_parser.add_argument("--representation-manifest", type=Path, required=True)
    evaluate_e02b_parser.add_argument("--cache-root", type=Path, required=True)
    evaluate_e02b_parser.add_argument("--scores", type=Path, required=True)
    evaluate_e02b_parser.add_argument("--selection", type=Path, required=True)
    evaluate_e02b_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_e02b_parser.add_argument("--condition", default="native")
    evaluate_e02b_parser.add_argument("--derivative-manifest", type=Path)
    evaluate_e02b_parser.add_argument("--derivative-root", type=Path)
    evaluate_e02b_parser.add_argument("--resolution-scores", type=Path)
    evaluate_e02b_parser.add_argument("--counterfactual-scores", type=Path, action="append")

    report_e02b = commands.add_parser(
        "generate-e02b-report",
        help="render the E02b report and 73-item traceability matrix",
    )
    report_e02b.add_argument("--config", type=Path, required=True)
    report_e02b.add_argument("--summary", type=Path, required=True)
    report_e02b.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a completed-phase CLI command and return a process exit code."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "build-manifest":
            patterns = ("**/*",) if arguments.all_files else arguments.include
            manifest = build_manifest(arguments.root, arguments.output, patterns)
            result = {
                "ok": True,
                "manifest": str(arguments.output.resolve()),
                "record_count": manifest["record_count"],
            }
        elif arguments.command == "verify-manifest":
            verification = verify_manifest(
                arguments.manifest,
                arguments.root,
                strict=arguments.strict,
            )
            result = verification.as_dict()
        elif arguments.command == "validate-prnu":
            result = validate_prnu(arguments.upstream_root, arguments.output)
        elif arguments.command == "validate-synthetic-fits":
            result = validate_synthetic_fits(
                arguments.config,
                arguments.output_dir,
                profile=arguments.profile,
            )
        elif arguments.command == "download-vision-e01":
            result = download_vision_e01(
                arguments.config,
                arguments.data_root,
                arguments.manifest,
            )
        elif arguments.command == "run-e01":
            result = run_e01(
                arguments.config,
                arguments.manifest,
                arguments.data_root,
                arguments.upstream_root,
                arguments.output_dir,
                arguments.cache_root,
            )
        elif arguments.command == "prepare-e02-data":
            result = prepare_e02_data(
                arguments.config,
                arguments.data_root,
                arguments.manifest,
            )
        elif arguments.command == "extract-residuals":
            result = extract_e02_residuals(
                arguments.config,
                arguments.manifest,
                arguments.data_root,
                arguments.upstream_root,
                arguments.cache_root,
                arguments.residual_manifest,
                extractor_names=arguments.extractor,
                noiseprint_root=arguments.noiseprint_root,
            )
        elif arguments.command == "build-fingerprint-bank":
            result = build_e02_fingerprint_bank(
                arguments.config,
                arguments.manifest,
                arguments.residual_manifest,
                arguments.bank_root,
                arguments.bank_manifest,
            )
        elif arguments.command == "score-pairs":
            result = score_e02_pairs(
                arguments.config,
                arguments.manifest,
                arguments.residual_manifest,
                arguments.bank_manifest,
                arguments.scores,
            )
        elif arguments.command == "evaluate-attribution":
            result = evaluate_e02(
                arguments.config,
                arguments.manifest,
                arguments.data_root,
                arguments.residual_manifest,
                arguments.scores,
                arguments.output_dir,
            )
        elif arguments.command == "generate-report":
            result = generate_e02_report(
                arguments.config,
                arguments.manifest,
                arguments.summary,
                arguments.evaluation,
                arguments.report,
            )
        elif arguments.command == "generate-e02b-shard":
            result = generate_e02b_shard(
                arguments.config,
                arguments.repository_root,
                arguments.data_root,
                arguments.cache_root,
                arguments.fragment,
                model_id=arguments.model_id,
                resolution=arguments.resolution,
                device=arguments.device,
                batch_size=arguments.batch_size,
            )
        elif arguments.command == "merge-e02b-generation":
            result = merge_e02b_fragments(
                arguments.config,
                arguments.repository_root,
                arguments.data_root,
                arguments.fragment_root,
                arguments.output,
            )
        elif arguments.command == "apply-e02b-counterfactuals":
            result = apply_e02b_counterfactuals(
                arguments.config,
                arguments.repository_root,
                arguments.generation_manifest,
                arguments.data_root,
                arguments.derivative_root,
                arguments.output,
            )
        elif arguments.command == "extract-e02b-representations":
            result = extract_e02b_representations(
                arguments.config,
                arguments.repository_root,
                arguments.generation_manifest,
                arguments.data_root,
                arguments.cache_root,
                arguments.fragment,
                arguments.upstream_root,
                extractor=arguments.extractor,
                condition=arguments.condition,
                derivative_manifest_path=arguments.derivative_manifest,
                derivative_root=arguments.derivative_root,
                noiseprint_root=arguments.noiseprint_root,
            )
        elif arguments.command == "merge-e02b-representations":
            result = merge_e02b_representation_fragments(
                arguments.config,
                arguments.repository_root,
                arguments.fragment_root,
                arguments.cache_root,
                arguments.output,
            )
        elif arguments.command == "score-e02b":
            result = score_e02b_matrix(
                arguments.config,
                arguments.repository_root,
                arguments.generation_manifest,
                arguments.data_root,
                arguments.representation_manifest,
                arguments.cache_root,
                arguments.scores,
                arguments.common_scores,
                arguments.calibration,
                condition=arguments.condition,
                derivative_manifest_path=arguments.derivative_manifest,
                derivative_root=arguments.derivative_root,
                profile=arguments.profile,
            )
        elif arguments.command == "select-e02b-condition":
            result = select_e02b_conditions(
                arguments.config,
                arguments.scores,
                arguments.output,
            )
        elif arguments.command == "evaluate-e02b":
            result = evaluate_e02b(
                arguments.config,
                arguments.repository_root,
                arguments.generation_manifest,
                arguments.data_root,
                arguments.representation_manifest,
                arguments.cache_root,
                arguments.scores,
                arguments.selection,
                arguments.output_dir,
                condition=arguments.condition,
                derivative_manifest_path=arguments.derivative_manifest,
                derivative_root=arguments.derivative_root,
                resolution_score_path=arguments.resolution_scores,
                counterfactual_score_paths=arguments.counterfactual_scores,
            )
        else:
            result = generate_e02b_report(
                arguments.config,
                arguments.summary,
                arguments.output_dir,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        succeeded = bool(result.get("ok", result.get("passed", False)))
        return 0 if succeeded else 1
    except (ImportError, ManifestError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
