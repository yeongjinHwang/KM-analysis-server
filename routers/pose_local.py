import mediapipe as mp
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
import aioboto3
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2

from utils.loader import yaml_loader, initialize_logger
from utils.detect import is_address, is_take_away, is_half
from utils.data_process import adaptive_ema

# YAML 설정 값 로드
CONFIG = yaml_loader("config.yaml")
KEYPOINTS = yaml_loader("keypoints.yaml")

S3_CONFIG = CONFIG.get("S3")
KEY_POINT_STRING = KEYPOINTS.get("key_point_string")
STRING_MATCH_INDEX = KEYPOINTS.get("string_match_index")
EIGHT_STEP = KEYPOINTS.get("eight_step")

class video_info(BaseModel):
    url: str  # ex) local video path
    handType: str  # ex) R, L

router = APIRouter()

# 사용자별 결과 저장
task_results = {}

# 비동기 FFmpeg 실행
async def run_ffmpeg_async(command):
    process = await asyncio.create_subprocess_exec(
        *command.split(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return stdout, stderr

async def mp_background(task_id, user_video_name, hand_type):
    """
    Mediapipe 모델을 백그라운드에서 실행.
    Args:
        task_id (str): 작업 ID.
        user_video_name (str): 다운로드된 비디오 파일 이름.
        hand_type (str): 손 타입 (R 또는 L).
    """
    # model init
    mp_pose = mp.solutions.pose
    full_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    heavy_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)

    # model running
    cap = cv2.VideoCapture(user_video_name)
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    frame: int = 0

    PoseLandMark = {joint: {"x": [], "y": [], "z_norm": [], "x_norm": []} for joint in KEY_POINT_STRING}
    none_frame = []
    is_swing = False
    detect_flow = "address"
    step = {}  # 자세 기록용
    address_count = 0  

    try: 
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            if frame == 0: height, width, channel = image.shape
            if hand_type == "L": image = cv2.flip(image, 1)

            # 모델 선택
            if is_swing == False : results = full_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else :  results = full_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # pose processing
            if not results.pose_landmarks:
                for joint in KEY_POINT_STRING:
                    # 이전 프레임 데이터로 채움, 이전 값이 없다면 0으로 초기화
                    PoseLandMark[joint]["x"].append(PoseLandMark[joint]["x"][-1] if frame > 0 else 0)
                    PoseLandMark[joint]["y"].append(PoseLandMark[joint]["y"][-1] if frame > 0 else 0)
                    PoseLandMark[joint]["z_norm"].append(PoseLandMark[joint]["z_norm"][-1] if frame > 0 else 0)
                    PoseLandMark[joint]["x_norm"].append(PoseLandMark[joint]["x_norm"][-1] if frame > 0 else 0)
                none_frame.append(frame)
            else:
                current_landmark = {}
                for joint, index in zip(KEY_POINT_STRING, STRING_MATCH_INDEX):
                    landmark = results.pose_landmarks.landmark[index]
                    world_landmark = results.pose_world_landmarks.landmark[index]
                    PoseLandMark[joint]["x"].append(landmark.x * width)  # 실제 픽셀 좌표 (x)
                    PoseLandMark[joint]["y"].append(landmark.y * height)  # 실제 픽셀 좌표 (y)
                    PoseLandMark[joint]["z_norm"].append(world_landmark.z)  # 정규화된 z 좌표
                    PoseLandMark[joint]["x_norm"].append(world_landmark.x)  # 정규화된 x 좌표
            
                    current_landmark[joint] = {
                        "x": landmark.x * width,
                        "y": landmark.y * height
                    }

                # 데이터 보정
                if is_swing==True :
                    for joint in KEY_POINT_STRING:
                        if frame > 0:
                            # 속도 계산
                            speed_x = abs(PoseLandMark[joint]["x"][-1] - PoseLandMark[joint]["x"][-2])
                            speed_y = abs(PoseLandMark[joint]["y"][-1] - PoseLandMark[joint]["y"][-2])
                            # Adaptive EMA 적용
                            PoseLandMark[joint]["x"][-1] = adaptive_ema(
                                PoseLandMark[joint]["x"][-2], PoseLandMark[joint]["x"][-1], speed_x
                            )
                            PoseLandMark[joint]["y"][-1] = adaptive_ema(
                                PoseLandMark[joint]["y"][-2], PoseLandMark[joint]["y"][-1], speed_y
                            ) 

                # detect
                match detect_flow :
                    case "address" :
                        if is_address(current_landmark):
                            address_count +=1
                            if address_count>=3 :
                                step["address"] = frame
                                is_swing=True
                                detect_flow = "take_away"
                        else :
                            address_count = 0
                        
                    case "take_away" :
                        if is_take_away(current_landmark) :
                            step["take_away"] = frame
                            detect_flow = "half"

                    case "half" :
                        if is_half(current_landmark) :
                            step["half"] = frame
                            detect_flow = "top" 

                    # case "top" :

                if detect_flow != "address" :
                    if is_address(current_landmark):
                        address_count +=1
                        if address_count>=3 :
                            step = {}
                            step["address"] = frame
                            is_swing=True
                            detect_flow = "take_away"
                    else :
                        address_count = 0
            frame += 1

        cap.release()

        task_results[task_id] = {"status": "completed", "pose": PoseLandMark, "step": step}

    except Exception as e:
        # 작업 실패 시 오류 저장
        task_results[task_id] = {"status": "error", "error": str(e)}

@router.post("/pose_local")
async def pose_local(request: video_info, background_tasks: BackgroundTasks): # mediapipe model자체는 동기
    """
    pose 추정 데이터 생성
    Args:
        request ``video_info``: 받은 요청 데이터 (비디오 경로 및 손 타입)
    Returns:
    """

    # request, config
    video_path, hand_type = request.url, request.handType

    # 20fps 변환
    ffmpeg_command = (
        f"ffmpeg -i {video_path} "
        f"-an -c:v libx264 "
        f"-r 20 "
        f"-vf \"setpts=N/(20*TB)\" "
        f"-y {video_path} -loglevel error"
    )
    await run_ffmpeg_async(ffmpeg_command)

    task_id = video_path

    # 백그라운드 작업 추가
    background_tasks.add_task(mp_background, task_id, video_path, hand_type)
    return {"task_id": task_id, "message": "Processing started"}
