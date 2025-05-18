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

    recommended_shapes = rules_face_brows[(brows_shape, face_shape)]
    recommended_types = rules_eye_spacing[eye_spacing]

    return [recommended_shapes, recommended_types]


def compute_P(selected_colors: list, selected_price_max, selected_brands: list, stars_color, stars_price, stars_brand, frames_color, frames_price, frames_brand):
    stars_sum = stars_price
    for selected_feature, stars_number in zip((selected_colors, selected_brands), (stars_color, stars_brand)):
        if selected_feature:
            stars_sum += stars_number
    star_value = 0.6/stars_sum
    P = 0
    if frames_price <= selected_price_max:
        P += stars_price * star_value

    for frames_feature, stars_number, selected_feature in zip((frames_color, frames_brand), (stars_color, stars_brand), (selected_colors, selected_brands)):
        if frames_feature in selected_feature:
            P += stars_number * star_value
    return P


def compute_score_per_frame(s_ai_impact_val, t_type_predicted, s_shape_predicted, t_type_selected, s_shape_selected,
                            frames_shape, frames_type, P):
    S = 0
    U = 0
    u_user_impact_val = 1 - s_ai_impact_val
    if t_type_predicted == frames_type:
        S += 0.5
    if s_shape_predicted == frames_shape:
        S += 0.5
    if t_type_selected == frames_type:
        U += 0.5
    if s_shape_selected == frames_shape:
        U += 0.5
    R = u_user_impact_val * U + s_ai_impact_val * S + P
    return R


if __name__ == '__main__':
    # get_recommendation('faces_examples\\face_example.jpg')
    P = compute_P(['blue', 'red'], 10, ['gucci', 'ck'],
                  2, 4, 8, 'red', 200, 'ck')
    print(P)
