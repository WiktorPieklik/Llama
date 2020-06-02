from pathlib import Path

SRC_DIR = Path(__file__).parent

SHAPE_PREDICTOR_MODEL_PATH = str(
    (
        SRC_DIR / "predictor_model.dat"
    ).resolve()
)

PREDICTOR_LANDMARKS = [
    0,
    2,
    8,
    14,
    16,
    17,
    19,
    21,
    22,
    24,
    26,
    27,
    31,
    33,
    35,
    36,
    37,
    39,
    41,
    42,
    44,
    45,
    46,
    48,
    50,
    52,
    54,
    57,
    62,
    66,
]
