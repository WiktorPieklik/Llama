import os

from cv2 import cv2

from face_detector import FaceDetector
from face_mask.points import FaceMaskPoints
from frame_source.camera import CameraFrameSource
from frame_source.source import ThreadedFrameSource
from shape_predictor import ShapePredictor

# from frame_source.video import VideoFrameSource

print(os.getcwd())
frame_source = CameraFrameSource(0)
# frame_source = VideoFrameSource("data/external/pexels_video/1.mp4")
# frame_source = ImageFrameSource(
#     "data/external/sof/AboA_00148_m_33_i_nf_nc_hp_2016_2_e0_nl_o.jpg"
# )
if isinstance(frame_source, ThreadedFrameSource):
    frame_source.start()

detector = FaceDetector()
predictor = ShapePredictor(
    "../data/external/dlib_models/shape_predictor_68_face_landmarks.dat"
)

mask = FaceMaskPoints(point_radius=1, point_color=(255, 255, 0))

print("Starting processing loop")
for frame in frame_source:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detect(gray)
    for face in faces:
        landmarks = predictor.predict(gray, face)
        mask.apply(frame, landmarks)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        if isinstance(frame_source, ThreadedFrameSource):
            frame_source.join()
        break
print("After processing loop")
