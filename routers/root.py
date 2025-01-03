from fastapi import APIRouter, Request
from utils.loader import initialize_logger

# 공통 로거 가져오기
logger = initialize_logger("app.log")

# APIRouter 생성
router = APIRouter()

@router.get("/", summary="Root Endpoint", description="서버 상태를 확인하는 기본 엔드포인트")
async def root(request: Request):
    """
    Root 엔드포인트 - 서버 상태를 확인
    """
    client_ip = request.client.host
    logger.info(f"Root 라우터 : {client_ip} -> 요청")
    return {"message": "Root endpoint is working!"}
