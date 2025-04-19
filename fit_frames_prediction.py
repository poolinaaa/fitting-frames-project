from prediction_brows import predict_brows
from prediction_face_shape import predict_shape
from eye_distance import check_eye_spacing


def get_recommendation(picture_path):
    eye_spacing = check_eye_spacing(picture_path)
    face_shape = predict_shape(picture_path)
    brows_shape = predict_brows(picture_path)
    rules_face_brows = {('round', 'heart'): [], ('round', 'oblong'): [], ('round', 'oval'): [], ('round', 'round'): [], ('round', 'square'): [],
                        ('high_arch', 'heart'): [], ('high_arch', 'oblong'): [], ('high_arch', 'oval'): [], ('high_arch', 'round'): [], ('high_arch', 'square'): [],
                        ('straight', 'heart'): [], ('straight', 'oblong'): [], ('straight', 'oval'): [], ('straight', 'round'): [], ('straight', 'square'): []}

    rules_eye_spacing = {'narrow': [], 'wide': [], 'balanced': []}

    recommended_models = rules_face_brows[(brows_shape, face_shape)]
    recommended_types = rules_eye_spacing[eye_spacing]

    return [recommended_models, recommended_types]


get_recommendation('faces_examples\\face_example.jpg')
