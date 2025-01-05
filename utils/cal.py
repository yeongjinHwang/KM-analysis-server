import math

def angle_xaxis(point1: list[float], point2: list[float]) -> float:
    """
    두 점을 잇는 선과 x축 사이의 각도를 계산하는 함수.

    Args:
        point1 ``list[float]``: 첫 번째 점의 [x, y] 좌표.
        point2 ``list[float]``: 두 번째 점의 [x, y] 좌표.

    Returns:
        float: x축과의 각도 (도 단위, 0~360도 사이).
    """
    # x, y 좌표 차이 계산
    delta_y = (1-point1[1])-(1-point2[1])
    delta_x = point1[0] - point2[0]
    
    # 기울기와 각도 계산
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    
    if angle_degrees < 0:
        angle_degrees += 360  # 음수일 경우 양수로 변환

    return angle_degrees

def center_point(point1: list[float], point2: list[float]) -> list[float]:
    """
    두 점의 중점 계산 함수.

    Args:
        point1 ``list[float]``: 첫 번째 점의 [x, y] 좌표.
        point2 ``list[float]``: 두 번째 점의 [x, y] 좌표.

    Returns:
        list[float]: 두 point의 중점
    """
    # x, y 좌표 차이 계산
    x = (point1[0] + point2[0])/2
    y = (point1[1] + point2[1])/2

    return [x,y]