from tensorflow.keras.models import load_model
import numpy as np
import pickle
from create_dataset_brows import load_dlib_detector, calculate_angles, detect_landmarks, preprocess_landmarks, visualize_detected_points, detect_landmarks


def prepare_data(pic):
    idxs_left = ((17, 18, 21), (17, 19, 21), (17, 20, 21),
                 (17, 18, 19), (17, 18, 20), (17, 19, 20))
    idxs_right = ((22, 23, 26), (22, 24, 26), (22, 25, 26),
                  (24, 25, 26), (23, 25, 26), (23, 24, 26))
    detector, predictor = load_dlib_detector()
    landmarks = detect_landmarks(pic, detector, predictor)
    #visualize_detected_points(pic, landmarks)
    dict_points_left, dict_points_right = preprocess_landmarks(landmarks)
    angles_left = calculate_angles(dict_points_left, idxs_left)
    angles_right = calculate_angles(dict_points_right, idxs_right)
    sample_data = np.array([angles_left + angles_right])
    return sample_data


#picture_path ex. 'faces_examples\\face_example.jpg'
# "models\\label_encoder_brows.pkl"
# "models\\scaler_brows.pkl"
# "models\\brows_model.h5"
def predict_brows(picture_path, model_brows="models\\brows_model.h5", label_encoder_brows="models\\label_encoder_brows.pkl", scaler_brows="models\\scaler_brows.pkl"):
    new_sample = prepare_data(picture_path)
    model = load_model(model_brows)

    with open(label_encoder_brows, "rb") as f:
        label_encoder = pickle.load(f)

    with open(scaler_brows, "rb") as f:
        scaler = pickle.load(f)

    new_sample_scaled = scaler.transform(new_sample)
    prediction = model.predict(new_sample_scaled)
    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])
    return predicted_label[0]

