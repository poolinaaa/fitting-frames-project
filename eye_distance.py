import cv2
import mediapipe as mp
import numpy as np
import os

def check_eye_spacing(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Plik nie istnieje: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Nie udało się wczytać obrazu: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils

    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return f"Nie wykryto twarzy na obrazie: {os.path.basename(image_path)}"

    for face_landmarks in results.multi_face_landmarks:

        left_eye_outer = face_landmarks.landmark[33]
        left_eye_inner = face_landmarks.landmark[133]
        right_eye_inner = face_landmarks.landmark[362]
        right_eye_outer = face_landmarks.landmark[263]

        # wspołrzędne landmarków
        height, width, _ = image.shape
        left_eye_inner_coords = np.array([int(left_eye_inner.x * width), int(left_eye_inner.y * height)])
        right_eye_inner_coords = np.array([int(right_eye_inner.x * width), int(right_eye_inner.y * height)])
        left_eye_outer_coords = np.array([int(left_eye_outer.x * width), int(left_eye_outer.y * height)])
        right_eye_outer_coords = np.array([int(right_eye_outer.x * width), int(right_eye_outer.y * height)])

        # wyliczanie średniej wielkości oczu i odległości między oczami
        left_eye_width = np.linalg.norm(left_eye_outer_coords - left_eye_inner_coords)
        right_eye_width = np.linalg.norm(right_eye_outer_coords - right_eye_inner_coords)
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        eye_distance = np.linalg.norm(left_eye_inner_coords - right_eye_inner_coords)

        # Klasyfikacja sprawdzająca jak się ma odległość między oczami do uśrednionej szerokości oczu
        ratio = eye_distance / avg_eye_width
        if ratio < 1:
            classification = "Wąski rozstaw oczu"
        elif ratio > 1.2:
            classification = "Szeroki rozstaw oczu"
        else:
            classification = "Zrównoważony rozstaw oczu"

        return {
            "classification": classification
        }

