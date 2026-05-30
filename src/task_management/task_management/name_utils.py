from __future__ import annotations

from typing import Dict, Iterable, Optional


CANONICAL_PARTS = [
    "flange nut",
    "gear ring",
    "spacer ring",
    "hex nut",
    "dom nut",
]

PART_ALIASES: Dict[str, str] = {
    "flange_nut": "flange nut",
    "flange nut": "flange nut",
    "flangenut": "flange nut",
    "플랜지 너트": "flange nut",
    "gear_ring": "gear ring",
    "gear ring": "gear ring",
    "gearring": "gear ring",
    "기어 링": "gear ring",
    "spacer_ring": "spacer ring",
    "spacer ring": "spacer ring",
    "spacerring": "spacer ring",
    "스페이서 링": "spacer ring",
    "hex_nut": "hex nut",
    "hex nut": "hex nut",
    "hexnut": "hex nut",
    "육각 너트": "hex nut",
    "dom_nut": "dom nut",
    "dom nut": "dom nut",
    "domnut": "dom nut",
    "dome_nut": "dom nut",
    "dome nut": "dom nut",
    "domenut": "dom nut",
    "돔 너트": "dom nut",
}

TRAY_ALIASES = {
    "blue_tray",
    "blue tray",
    "bluetray",
    "tray_blue",
    "tray blue",
    "blue_bin",
    "blue bin",
    "파란색 트레이",
    "파란 트레이",
}


def normalize_key(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("-", "_").split())


def compact_key(name: str) -> str:
    return normalize_key(name).replace("_", " ")


def canonical_part_name(name: str) -> Optional[str]:
    key = normalize_key(name)
    if key in PART_ALIASES:
        return PART_ALIASES[key]

    key = compact_key(name)
    return PART_ALIASES.get(key)


def is_tray_name(name: str, tray_names: Iterable[str] = TRAY_ALIASES) -> bool:
    keys = {normalize_key(v) for v in tray_names}
    keys.update(compact_key(v) for v in tray_names)
    return normalize_key(name) in keys or compact_key(name) in keys
