"""Allow ``python -m gfits`` to invoke the command-line interface."""

from gfits.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
