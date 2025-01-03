import yaml
import logging
from typing import Any

def yaml_loader(path: str) -> dict[str, Any]:
    """
    YAML 파일 로더
    Args:
        path (str): YAML 파일 경로
    Returns:
        dict[str, Any]: 로드된 YAML 파일의 데이터
    Raises:
        FileNotFoundError: YAML 파일이 존재하지 않는 경우
        ValueError: YAML 파일 로드 중 오류가 발생한 경우
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 파일 로드 중 에러가 발생했습니다: {e}") from e
    
# 공통 로거
logger = None
def initialize_logger(log_file: str) -> logging.Logger:
    """
    로거 초기화 및 설정
    Args:
        log_file ``str``: 로그 파일 경로
    Returns:
        logging.Logger: 초기화된 로거
    """
    global logger
    if logger is not None:
        return logger

    logger = logging.getLogger("shared_logger")
    
    # 중복 핸들러 방지
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.setLevel(logging.INFO)  # 로그 레벨 설정
        logger.addHandler(file_handler)  # 파일 핸들러 추가

    return logger