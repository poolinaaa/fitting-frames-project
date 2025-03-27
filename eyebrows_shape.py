import cv2
import dlib
import numpy as np
import glob
import os
import re
import pandas as pd

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
    if image is None:
        print(f"Couldn't load image: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print(f"Couldn't detect face in the picture: {image_path}")
        return None

    
    landmarks = predictor(gray, faces[0])
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

    cv2.imshow("Eyebrows detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_df_for_class(path):
    idxs_left = ((17, 18, 21), (17, 19, 21), (17, 20, 21),
             (17, 18, 19), (17, 18, 20), (17, 19, 20))
    idxs_right = ((22, 23, 26), (22, 24, 26), (22, 25, 26),
              (24, 25, 26), (23, 25, 26), (23, 24, 26))
    samples_calculated = []
    detector, predictor = load_dlib_detector()
    
    
    search_path = os.path.join(path, "*.*")
    for pic in glob.glob(search_path):
        print(pic)
        match = re.search(r'\\([^\\]+)', pic)
        
        landmarks = detect_landmarks(pic, detector, predictor)
        if landmarks is None:
            continue
        dict_points_left, dict_points_right = preprocess_landmarks(landmarks)
        angles_left = calculate_angles(dict_points_left, idxs_left)
        angles_right = calculate_angles(dict_points_right, idxs_right)
        
        sample_data = angles_left + angles_right + [str(match.group(1))]
        samples_calculated.append(sample_data)
    df = pd.DataFrame(samples_calculated,columns=['l_1','l_2','l_3','l_4','l_5','l_6','p_1','p_2','p_3','p_4','p_5','p_6','brows_class'])   
    return df





if __name__=="__main__":
    classes = ['round', 'high_arch', 'straight']
    df_brows = pd.DataFrame()
    for brows_type in classes:
    
        df = create_df_for_class(f'brows\\{brows_type}')
        df_brows = pd.concat([df_brows, df], ignore_index=True)
    print(df_brows.head)
    df_brows.to_csv('brows_raw.csv', index = True) 