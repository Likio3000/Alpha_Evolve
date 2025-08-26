from __future__ import annotations
import argparse
from dataclasses import fields as dc_fields
import argparse


def add_dataclass_args(
    parser: argparse.ArgumentParser,
    dc_type,
    *,
    skip: set[str] | None = None,
    choices_map: dict[str, list[str]] | None = None,
    already_added: set[str] | None = None,
) -> set[str]:
    """Add argparse flags for simple dataclass fields.

    - Supports int/float/str/bool fields.
    - Uses ``default=argparse.SUPPRESS`` so dataclass defaults apply unless provided.
    - Returns the set of field names added, useful to avoid duplicates across dataclasses.
    """
    skip = skip or set()
    choices_map = choices_map or {}
    added = set() if already_added is None else set(already_added)
    for f in dc_fields(dc_type):
        name = f.name
        if name in skip or name in added:
            continue
        ftype = f.type
        if ftype not in (int, float, str, bool):
            continue
        if name == "generations":
            # Keep positional form for compatibility where relevant
            continue
        arg = f"--{name}"
        kwargs: dict = {"default": argparse.SUPPRESS}
        if ftype is bool:
            # Primary flag sets True
            parser.add_argument(arg, dest=name, action="store_true", **kwargs)
            # If default is True, also provide a --no-<name> to set False
            default_val = getattr(dc_type, name, None)
            try:
                # Try to get default from dataclass instance if attribute exists
                default_val = getattr(dc_type(), name)  # type: ignore
            except Exception:
                pass
            if default_val is True:
                parser.add_argument(f"--no-{name}", dest=name, action="store_false", default=argparse.SUPPRESS)
        else:
            kwargs["type"] = ftype
            if name in choices_map:
                kwargs["choices"] = choices_map[name]
            parser.add_argument(arg, **kwargs)
        added.add(name)
    return added
