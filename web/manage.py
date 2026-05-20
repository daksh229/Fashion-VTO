#!/usr/bin/env python
"""Django CLI entry. Run from the `web/` directory."""
import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Make sure the venv is activated "
            "and `Django` is installed."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
