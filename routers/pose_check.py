from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/pose_check")
async def get_all_task_results(app: Request):
    """
    모든 task_results 반환.
    """
    return app.app.state.task_results