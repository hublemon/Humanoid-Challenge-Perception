"""Shared constants for A_command parts readers."""

PART_CLASS_NAMES = [
    "flange_nut",
    "gear_ring",
    "spacer_ring",
    "hex_nut",
    "dome_nut",
]

PART_CLASS_TO_NAME = {
    "flange_nut": "플랜지 너트",
    "gear_ring": "기어 링",
    "spacer_ring": "스페이서 링",
    "hex_nut": "육각 너트",
    "dome_nut": "돔 너트",
}

PART_NAMES = [PART_CLASS_TO_NAME[class_name] for class_name in PART_CLASS_NAMES]
N_ROWS = len(PART_NAMES)
VALID_DIGITS = tuple(range(6))
