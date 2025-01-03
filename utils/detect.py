import math

def cal_angle_xaxis(point1, point2) -> float:
    """
    두 점을 잇는 선과 x축 사이의 각도를 계산하는 함수.

    Args:
        point1 ``list[float]``: 첫 번째 점의 [x, y] 좌표.
        point2 ``list[float]``: 두 번째 점의 [x, y] 좌표.

    Returns:
        float: x축과의 각도 (도 단위, 0~360도 사이).
    """
    # x, y 좌표 차이 계산
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    
    # 기울기와 각도 계산
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    
    if angle_degrees < 0:
        angle_degrees += 360  # 음수일 경우 양수로 변환

    return angle_degrees

def is_address(current_landmark) -> bool:
    """
    골프 어드레스 자세를 감지하는 함수.

    Args:
        current_landmark ``dict[str, list[float]]``: Mediapipe 포즈 랜드마크 데이터(하나의 frame).

    Returns:
        bool: 어드레스 자세가 감지되면 True, 그렇지 않으면 False.
    """
    # 주요 랜드마크 좌표
    right_wrist = current_landmark["right_wrist"]
    right_hip = current_landmark["right_hip"]
    right_knee = current_landmark["right_knee"]
    right_shoulder = current_landmark["right_shoulder"]
    right_hip = current_landmark["right_hip"]

    # 1. 상체 기울기 계산
    spine_angle = cal_angle_xaxis(right_shoulder, right_hip)

    # 상체 기울기가 적절한지 확인
    if not (10 <= spine_angle <= 70):
        return False

    # 2. 손목 위치 조건
    wrist_y = right_wrist[1]
    hip_y = right_hip[1]
    knee_y = right_knee[1]

    if not (hip_y < wrist_y < knee_y): # 엉덩이보다는 낮고 무릎보다는 높아야됨
        return False
    
    return True