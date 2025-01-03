def adaptive_ema(prev_value, current_value, speed, alpha_base=0.2, speed_threshold=10):
    """
    적응형 지수 가중 이동 평균
    Args:
        prev_value: 이전 값
        current_value: 현재 값
        speed: 현재 속도 값
        alpha_base: 기본 alpha 값
        speed_threshold: 속도 임계값
    Returns:
        스무딩된 값
    """
    # 속도가 빠르면 alpha 값을 증가
    alpha = alpha_base + min(0.3, speed / speed_threshold)
    return alpha * current_value + (1 - alpha) * prev_value