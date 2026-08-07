"""Command-line entry point for completed G-FITS phases."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gfits import __version__
from gfits.manifest import ManifestError, build_manifest, verify_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gfits",
        description="G-FITS reproducible research utilities (Phase 0)",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Phase 0 CLI and return a process exit code."""

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
        else:
            verification = verify_manifest(
                arguments.manifest,
                arguments.root,
                strict=arguments.strict,
            )
            result = verification.as_dict()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    except (ManifestError, OSError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
