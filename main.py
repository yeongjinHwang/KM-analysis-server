from fastapi import FastAPI
import subprocess
from utils.loader import yaml_loader, initialize_logger

from routers.root import router as root
from routers.pose import router as pose

# 로그 초기화
LOG_FILE = "app.log"
logger = initialize_logger(LOG_FILE)

# 설정 파일 로드
settings = yaml_loader("config.yaml")

# 필수 설정 로드
IP_NUM: str = settings.get("IP_NUM", "127.0.0.1")
PORT_NUM: str = settings.get("PORT_NUM", "8000")
WORKER: str = settings.get("WORKERS", "5")

# FastAPI 애플리케이션 생성
app = FastAPI()

# 라우터 등록
try:
    app.include_router(root)
    app.include_router(pose)
except NameError as e:
    logger.error(f"Root 라우터 등록 중 오류 발생: {e}")
    raise NameError("라우터가 정의되지 않았습니다..") from e

# 메인 실행부
if __name__ == "__main__":
    logger.info("FastAPI 서버 시작 준비 중...")
    try:
        # subprocess 명령어 생성
        command = [
            "uvicorn",
            "main:app",
            "--host", IP_NUM,
            "--port", PORT_NUM,
            "--workers", WORKER,
        ]

        # 실행 명령어를 로그에 기록
        logger.info(f"서버 실행 명령어: {' '.join(command)}")

        # 명령어 실행
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"서버 실행 중 오류가 발생했습니다: {e}")
        raise RuntimeError(f"서버 실행 중 오류가 발생했습니다: {e}") from e
