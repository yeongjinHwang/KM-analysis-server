from fastapi import FastAPI
import subprocess
from utils.loader import yaml_loader

from routers.root import router as root
from routers.pose import router as pose

#test
from routers.pose_local import router as pose_local
from routers.pose_check import router as pose_check

# 설정 파일 로드
settings = yaml_loader("config.yaml")

# 필수 설정 로드
IP_NUM: str = settings.get("IP_NUM", "127.0.0.1")
PORT_NUM: str = settings.get("PORT_NUM", "8000")
WORKER: str = settings.get("WORKERS", "5")

# FastAPI 애플리케이션 생성
app = FastAPI()
# 전역 상태 초기화
app.state.task_results = {}

# 라우터 등록
app.include_router(root)
app.include_router(pose)
app.include_router(pose_local)
app.include_router(pose_check)

# 메인 실행부
if __name__ == "__main__":
    try:
        # subprocess 명령어 생성
        command = [
            "uvicorn",
            "main:app",
            "--host", IP_NUM,
            "--port", PORT_NUM,
            "--workers", WORKER,
        ]

        # 명령어 실행
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"서버 실행 중 오류가 발생했습니다: {e}")
        raise RuntimeError(f"서버 실행 중 오류가 발생했습니다: {e}") from e
