import cv2
import dlib

from face_mask.points import FaceMaskPoints

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "../data/external/dlib_models/shape_predictor_68_face_landmarks.dat"
)

mask = FaceMaskPoints(point_radius=2, point_color=(0, 255, 255))

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        # x1 = face.left()
        # y1 = face.top()
        # x2 = face.right()
        # y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray, face)
        mask.apply(frame, landmarks)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
