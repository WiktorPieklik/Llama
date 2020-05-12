from pathlib import Path

SRC_DIR = Path(__file__).parent

SHAPE_PREDICTOR_MODEL_PATH = str(
    (
        SRC_DIR / "../data/external/dlib_models/shape_predictor_68_face_landmarks.dat"
    ).resolve()
)

