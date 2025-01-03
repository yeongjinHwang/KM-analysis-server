import math

def cal_angle_xaxis(point1: list[float], point2: list[float]) -> float:
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

def is_address(current_landmark: dict[str, dict[str, float]]) -> bool:
    """
    골프 어드레스 자세를 감지하는 함수.

    Args:
        current_landmark ``dict[str, list[float]]``: Mediapipe 포즈 랜드마크 데이터(현재 프레임).

    Returns:
        bool: 어드레스 자세가 감지되면 True, 그렇지 않으면 False.
    """
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

def is_take_away(current_landmark: dict[str, dict[str, float]]) -> bool:
    """
    골프 테이크어웨이 자세를 감지하는 함수.
    Args:
        current_landmark ``dict[str, list[float]]``: Mediapipe 포즈 랜드마크 데이터(현재 프레임).
    Returns:
        bool: 테이크어웨이 자세가 감지되면 True, 그렇지 않으면 False.
    """
    left_hip = current_landmark["left_hip"]
    right_hip = current_landmark["right_hip"]
    right_wrist = current_landmark["right_wrist"]

    # 1. 손목의 y 좌표가 엉덩이의 y 좌표 이상인지 확인
    # 오른손목이 왼쪽 엉덩이보다 아래에 있으면 False
    # 오른손목이 오른쪽 엉덩이보다 아래에 있으면 False
    if right_wrist["y"] > left_hip["y"] or right_wrist["y"] > right_hip["y"]:
        return False

    return True

def is_half(current_landmark: dict[str, dict[str, float]]) -> bool:
    """
    골프 스윙의 Half 자세를 감지하는 함수.
    Args:
        current_landmark ``dict[str, dict[str, float]]``: Mediapipe 포즈 랜드마크 데이터(현재 프레임).
    Returns:
        bool: Half 자세가 감지되면 True.
    """
    # 주요 랜드마크 좌표
    left_shoulder = current_landmark["left_shoulder"]
    right_shoulder = current_landmark["right_shoulder"]
    left_wrist = current_landmark["left_wrist"]
    right_wrist = current_landmark["right_wrist"]
    left_elbow = current_landmark["left_elbow"]
    right_elbow = current_landmark["right_elbow"]

    # 1. 어깨의 y 좌표 중 최대값 계산 (화면에서 아래쪽으로 내려갈수록 y 값이 커짐)
    shoulder_y = max(left_shoulder["y"], right_shoulder["y"])

    # 2. 손목 및 팔꿈치 y 좌표를 어깨와 비교하여 조건 확인
    y_values = [right_wrist["y"], left_wrist["y"], right_elbow["y"], left_elbow["y"]]
    if sum(y <= shoulder_y for y in y_values) < 2:
        return False

    # 모든 조건을 만족하지 못한 경우에도 기본적으로 True 반환
    return True

def is_top(current_landmark: dict[str, dict[str, float]]) -> bool:
    """
    골프 스윙의 Top 자세를 감지하는 함수.
    Args:
        current_landmark ``dict[str, dict[str, float]]``: Mediapipe 포즈 랜드마크 데이터(현재 프레임).
    Returns:
        bool: Half 자세가 감지되면 True.
    """
