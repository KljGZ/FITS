"""Command-line entry point for completed G-FITS phases."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gfits import __version__
from gfits.manifest import ManifestError, build_manifest, verify_manifest
from gfits.prnu_validation import validate_prnu
from gfits.synthetic import validate_synthetic_fits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gfits",
        description="G-FITS reproducible research utilities (Phase 0 and E00)",
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
        else:
            result = validate_synthetic_fits(
                arguments.config,
                arguments.output_dir,
                profile=arguments.profile,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        succeeded = bool(result.get("ok", result.get("passed", False)))
        return 0 if succeeded else 1
    except (ImportError, ManifestError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
