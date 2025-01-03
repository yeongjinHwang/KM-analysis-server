import mediapipe as mp
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import boto3
from botocore.config import Config
import os
import cv2

from utils.loader import yaml_loader, initialize_logger
from utils.detect import is_address

# YAML 설정 값 로드
CONFIG = yaml_loader("config.yaml")
KEYPOINTS = yaml_loader("keypoints.yaml")

S3_CONFIG = CONFIG.get("S3")
KEY_POINT_STRING = KEYPOINTS.get("key_point_string")
STRING_MATCH_INDEX = KEYPOINTS.get("string_match_index")
EIGHT_STEP = KEYPOINTS.get("eight_step")


class video_info(BaseModel):
    url: str  # ex) username/videoname.mp4
    handType: str  # ex) R, L
router = APIRouter()

@router.post("/pose")
async def pose(request: video_info):
    """
    pose 추정 데이터 생성
    Args:
        request ``video_info``: 받은 요청 데이터 (비디오 경로 및 손 타입)
    Returns:
    """

    # request, config
    video_path, hand_type = request.url, request.handType
    user_video_name = video_path.replace("/", "_")  # ex) username/videoname.mp4 -> username_videoname.mp4

    # Boto3 세션 및 S3 객체 초기화
    session = boto3.Session(
        aws_access_key_id=S3_CONFIG.get("s3_accesskey"),
        aws_secret_access_key=S3_CONFIG.get("s3_privatekey"),
        region_name=S3_CONFIG.get("s3_region_name")
    )
    s3 = session.resource('s3')
    object = s3.Object(S3_CONFIG.get("s3_bucket_name"), video_path)
    # Boto3 Config 설정
    boto_config = Config(
        connect_timeout=10,  # 최대 timeout 시간
        read_timeout=30  # 최대 read 시간
    )
        
    # model init
    mp_pose = mp.solutions.pose
    full_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    heavy_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)

    # video download
    object.download_file(user_video_name, Config=boto_config)

    # video 20fps transform
    input_video = user_video_name
    command = (
        f"ffmpeg -i {input_video} "
        f"-an -c:v libx264 "
        f"-r 20 "
        f"-vf \"setpts=N/(20*TB)\" "
        f"-y {input_video} -loglevel error"
    )
    os.system(command)

    # model running
    cap = cv2.VideoCapture(user_video_name)
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    frame: int = 0

    PoseLandMark = {joint: {"x": [], "y": [], "z_norm": [], "x_norm": []} for joint in KEY_POINT_STRING}
    none_frame = []
    is_swing = False
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        if frame == 0: height, width, channel = image.shape

        # 손 타입에 따라 이미지 플립
        if hand_type == "L": image = cv2.flip(image, 1)

        # Mediapipe Full Model 실행
        if is_swing == False :
            results = full_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else : 
            results = full_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 포즈 데이터 처리
        if not results.pose_landmarks:
            for joint in KEY_POINT_STRING:
                # 이전 프레임 데이터로 채움, 이전 값이 없다면 0으로 초기화
                PoseLandMark[joint]["x"].append(PoseLandMark[joint]["x"][-1] if frame > 0 else 0)
                PoseLandMark[joint]["y"].append(PoseLandMark[joint]["y"][-1] if frame > 0 else 0)
                PoseLandMark[joint]["z_norm"].append(PoseLandMark[joint]["z_norm"][-1] if frame > 0 else 0)
                PoseLandMark[joint]["x_norm"].append(PoseLandMark[joint]["x_norm"][-1] if frame > 0 else 0)
            none_frame.append(frame)
        else:
             # 현재 프레임 데이터를 저장
            current_landmark = {}
            for joint, index in zip(KEY_POINT_STRING, STRING_MATCH_INDEX):
                landmark = results.pose_landmarks.landmark[index]
                world_landmark = results.pose_world_landmarks.landmark[index]
                PoseLandMark[joint]["x"].append(landmark.x * width)  # 실제 픽셀 좌표 (x)
                PoseLandMark[joint]["y"].append(landmark.y * height)  # 실제 픽셀 좌표 (y)
                PoseLandMark[joint]["z_norm"].append(world_landmark.z)  # 정규화된 z 좌표
                PoseLandMark[joint]["x_norm"].append(world_landmark.x)  # 정규화된 x 좌표
        
                # 현재 프레임 데이터 저장
                current_landmark[joint] = {
                    "x": landmark.x * width,
                    "y": landmark.y * height
                }
            # 어드레스 자세 감지
            if is_address(current_landmark):
                is_swing=True
                print(f"Frame {frame}: Address detected!")

        frame += 1

    cap.release()