import math
from utils.cal import angle_xaxis, center_point

def is_address(current_landmark: dict[str, dict[str, float]], landmark: dict[str, dict[str, float]]) -> bool:
    """
    골프 어드레스 자세를 감지하는 함수.

    Args:
        current_landmark ``dict[str, list[float]]`` : Mediapipe 포즈 랜드마크 데이터(현재 프레임).
        landmark ``dict[str, dict[str, float]]`` : Mediapipe 포즈 랜드마크 데이터

    Returns:
        bool: 어드레스 자세가 감지되면 True, 그렇지 않으면 False.
    """
    # 각 부위의 좌표 추출
    right_wrist = current_landmark["right_wrist"]
    right_hip = current_landmark["right_hip"]
    right_knee = current_landmark["right_knee"]
    right_shoulder = current_landmark["right_shoulder"]

    prev_right_wrist_x = landmark["right_wrist"]["x"][-2:] # 마지막 2개
    prev_right_wrist_y = landmark["right_wrist"]["y"][-2:] 


    # 상체 기울기 계산
    spine_angle = angle_xaxis(
        [right_shoulder["x"], right_shoulder["y"]],
        [right_hip["x"], right_hip["y"]]
    )
    if not (10 <= spine_angle <= 70):
        return False

    # 손목 위치 조건 확인
    if not (right_hip["y"] < right_wrist["y"] < right_knee["y"]):
        return False
    

    # x 좌표 차이 확인
    x_tolerance_check = all(
        abs(right_wrist["x"] - prev_x) / max(abs(right_wrist["x"]), 1e-6) <= 0.01 for prev_x in prev_right_wrist_x
    )
    # y 좌표 차이 확인
    y_tolerance_check = all(
        abs(right_wrist["y"] - prev_y) / max(abs(right_wrist["y"]), 1e-6) <= 0.01 for prev_y in prev_right_wrist_y
    )

    # 손목이 엉덩이 아래, 무릎 위에 있어야 함, 척추각도가 10~70도 사이에 있어야함, x,y 모두 1%이내 차이 연속3frame
    return x_tolerance_check and y_tolerance_check

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

    # 손목이 왼-오엉덩이보다 높게 있어야함
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

    # 손및 및 팔꿈치 4개중 낮은어깨보다 위에있는게 2개이상
    return True

def is_top(current_landmark: dict[str, dict[str, float]], landmark: dict[str, dict[str, float]], step: int) -> bool:
    """
    골프 스윙의 Top 자세를 감지하는 함수.
    Args:
        current_landmark ``dict[str, dict[str, float]]``: Mediapipe 포즈 랜드마크 데이터(현재 프레임).
        landmark  ``dict[str, dict[str, float]]``: Mediapipe 포즈 랜드마크 데이터
        step ``int`` : sequence
    Returns:
        bool: 현재 손이 최고점값이면 True 아니면 False
    """

    # landmark 데이터에서 y-좌표만 비교
    right_wrist_y_values = landmark["right_wrist"]["y"][step:]
    current_y = current_landmark["right_wrist"]["y"]

    if current_y != min(right_wrist_y_values):
        return False
    
    # 현재 오른손목 y좌표가 이전 y좌표 중 최고점이어야함
    return True

def is_down_half(current_landmark: dict[str, dict[str, float]]) -> bool:
    """
    골프 스윙의 Down Half 자세를 감지하는 함수.
    Args:
        current_landmark ``dict[str, dict[str, float]]``: Mediapipe 포즈 랜드마크 데이터(현재 프레임).
    Returns:
        bool: Down Half 자세가 감지되면 True, 아니면 False.
    """
    # 주요 랜드마크 좌표
    right_shoulder = current_landmark["right_shoulder"]
    right_wrist = current_landmark["right_wrist"]

    if right_wrist["y"] <= right_shoulder["y"] :
        return False
    
    # 손목이 어깨보다 낮게 있어야함
    return True

def is_impact(current_landmark: dict[str, dict[str, float]], landmark: dict[str, dict[str, float]], step: int) -> bool:
    """
    골프 스윙의 Impact 자세를 감지하는 함수.
    Args:
        current_landmark ``dict[str, dict[str, float]]``: Mediapipe 포즈 랜드마크 데이터(현재 프레임).
        landmark ``dict[str, dict[str, float]]``: Mediapipe 포즈 랜드마크 데이터
        step ``int``: sequence

    Returns:
        bool: 현재 손이 최저점이면 True 아니면 False.
    """

    # y-좌표 데이터 비교
    right_wrist_y_values = right_wrist_y_values = landmark["right_wrist"]["y"][step:]
    current_y = current_landmark["right_wrist"]["y"]

    if current_y != max(right_wrist_y_values):
        return False

    # 현재 오른손목 y좌표가 이전 y좌표 중 최저점이어야함
    return True

def is_follow_through(current_landmark: dict[str, dict[str, float]]) -> bool :
    """
    골프 스윙의 follow_through 자세를 감지하는 함수.
    Args:
        current_landmark ``dict[str, dict[str, float]]``: Mediapipe 포즈 랜드마크 데이터(현재 프레임).
    Returns:
        bool: follow through 자세가 감지되면 True, 아니면 False.
    """

    # 각 부위의 좌표 추출
    right_wrist = current_landmark["right_wrist"]
    right_shoulder = current_landmark["right_shoulder"]
    left_shoulder = current_landmark["left_shoulder"]
    right_hip = current_landmark["right_hip"]
    left_hip = current_landmark["left_hip"]

    shoulder_y = center_point(
        [right_shoulder["x"], right_shoulder["y"]],
        [left_shoulder["x"], left_shoulder["y"]]
    )[1]

    hip_y = center_point(
        [right_hip["x"], right_hip["y"]],
        [left_hip["x"], left_hip["y"]]
    )[1] 

    base_y = (hip_y - abs(shoulder_y-hip_y)/3)
    if base_y <= right_wrist["y"] : 
        return False
    
    # 손목이 엉덩이-어깨 3등분점보다 높아야함
    return True

def is_finish(current_landmark: dict[str, dict[str, float]], landmark: dict[str, dict[str, float]], step: int) -> bool :
    """
    골프 스윙의 finish 자세를 감지하는 함수.
    Args:
        current_landmark ``dict[str, dict[str, float]]``: Mediapipe 포즈 랜드마크 데이터(현재 프레임).
        landmark ``dict[str, dict[str, float]]``: Mediapipe 포즈 랜드마크 데이터
        step ``int`` : sequence
    Returns:
        bool: finish 자세가 감지되면 True, 아니면 False.
    """
    
    right_wrist = current_landmark["right_wrist"]

    base = center_point(
        [landmark["left_shoulder"]["x"][step], landmark["left_shoulder"]["y"][step]],
        [landmark["left_wrist"]["x"][step], landmark["left_wrist"]["y"][step]]
    )

    # 2:1비율로
    base_y = center_point(
        base,
        [landmark["left_wrist"]["x"][step], landmark["left_wrist"]["y"][step]]
    )[1]

    if base_y>=right_wrist["y"] :
        return False
    
    # 기준선보다 손이 낮아야됨
    return True