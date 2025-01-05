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