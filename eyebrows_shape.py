import cv2
import dlib
import numpy as np


def load_dlib_detector():
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    return detector, predictor


def calculate_angle(point1, vertex, point2):

    vector1 = np.array([point1[0] - vertex[0], point1[1] - vertex[1]])
    vector2 = np.array([point2[0] - vertex[0], point2[1] - vertex[1]])

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (norm1 * norm2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_angles(dict_points, idxs_of_points):
    calculated_angles = []
    for set_idxs in idxs_of_points:
        point1 = dict_points[set_idxs[0]]
        vertex = dict_points[set_idxs[1]]
        point2 = dict_points[set_idxs[2]]
        angle = calculate_angle(point1, vertex, point2)
        calculated_angles.append(angle)
    return calculated_angles


def detect_landmarks(image_path, detector, predictor):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
    return landmarks


def preprocess_landmarks(landmarks):
    left_eyebrow = {i: (landmarks.part(i).x, landmarks.part(i).y)
                    for i in range(17, 22)}
    right_eyebrow = {i: (landmarks.part(i).x, landmarks.part(i).y)
                     for i in range(22, 27)}

    return left_eyebrow, right_eyebrow


def visualize_detected_points(image_path, landmarks):
    image = cv2.imread(image_path)

    left_eyebrow = list(range(17, 22))
    right_eyebrow = list(range(22, 27))

    for i in left_eyebrow + right_eyebrow:
        x, y = landmarks.part(i).x, landmarks.part(i).y

        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Eyebrows Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = 'face_example.jpg'
detector, predictor = load_dlib_detector()
landmarks = detect_landmarks(img, detector, predictor)
dict_points_left, dict_points_right = preprocess_landmarks(landmarks)
idxs_left = ((17, 18, 21), (17, 19, 21), (17, 20, 21),
             (17, 18, 19), (17, 18, 20), (17, 19, 20))
idxs_right = ((22, 23, 26), (22, 24, 26), (22, 25, 26),
              (24, 25, 26), (23, 25, 26), (23, 24, 26))
angles_left = calculate_angles(dict_points_left, idxs_left)
print(angles_left)
print(dict_points_left)
visualize_detected_points(img, landmarks)
