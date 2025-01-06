import mediapipe as mp
from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel
import cv2
import os 

from utils.loader import yaml_loader
from utils.detect import is_address, is_take_away, is_half, is_top, is_down_half, is_impact, is_follow_through, is_finish
from utils.data_process import adaptive_ema

# YAML 설정 값 로드
KEYPOINTS = yaml_loader("keypoints.yaml")

KEY_POINT_STRING = KEYPOINTS.get("key_point_string")
STRING_MATCH_INDEX = KEYPOINTS.get("string_match_index")
EIGHT_STEP = KEYPOINTS.get("eight_step")

class video_info(BaseModel):
    url: str  # ex) local video path
    handType: str  # ex) R, L

router = APIRouter()

async def mp_background(task_id, user_video_name, hand_type, task_results):
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

    # 결과 이미지 저장 폴더 생성
    image_dir = f"images/{task_id}/frame"
    os.makedirs(image_dir, exist_ok=True)
    step_dir = f"images/{task_id}/step"
    os.makedirs(step_dir, exist_ok=True)

    PoseLandMark = {joint: {"x": [], "y": [], "z_norm": [], "x_norm": []} for joint in KEY_POINT_STRING}
    none_frame = []
    is_swing = False
    detect_flow = "address"
    step = {}  # 자세 기록용

    try: 
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            if frame == 0: height, width, channel = image.shape
            if hand_type == "L": image = cv2.flip(image, 1)

            # 모델 선택
            if is_swing == False : results = full_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else :  results = heavy_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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

                current_landmark = {
                    joint: {
                        "x": PoseLandMark[joint]["x"][-1],
                        "y": PoseLandMark[joint]["y"][-1],
                    }
                    for joint in KEY_POINT_STRING
                }

                # 보정된 좌표 기준으로 원(circle) 그리기
                for joint, coord in current_landmark.items():
                    x, y = int(coord["x"]), int(coord["y"])
                    cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

                # 이미지 저장
                output_path_frame = os.path.join(image_dir, f"{frame}.png")
                cv2.imwrite(output_path_frame, image)

            output_path_step = os.path.join(step_dir, f"{detect_flow}.png")
            # detect
            match detect_flow :
                case "address":
                    if frame>=3 :
                        if is_address(current_landmark, PoseLandMark):
                            step["address"] = frame
                            is_swing=True
                            cv2.imwrite(output_path_step, image)
                            detect_flow = "take_away"

                case "take_away" :
                    if is_take_away(current_landmark) :
                        step["take_away"] = frame
                        cv2.imwrite(output_path_step, image)
                        detect_flow = "half"

                case "half" :
                    if is_half(current_landmark) :
                        step["half"] = frame
                        cv2.imwrite(output_path_step, image)
                        detect_flow = "top" 
                
                case "top" :
                    if is_top(current_landmark, PoseLandMark, step["half"]) :
                        step["top"] = frame
                        cv2.imwrite(output_path_step, image)
                    else :
                        detect_flow = "donw_half"
                
                case "donw_half" :
                    if is_down_half(current_landmark) :
                        step["down_half"] = frame
                        cv2.imwrite(output_path_step, image)
                        detect_flow = "impact"

                case "impact" :
                    if is_impact(current_landmark, PoseLandMark, step["down_half"]) :
                        step["impact"] = frame
                        cv2.imwrite(output_path_step, image)
                    else :
                        detect_flow = "follow_through"

                case "follow_through" :
                    if is_follow_through(current_landmark) :
                        step["follow_through"] = frame
                        cv2.imwrite(output_path_step, image)
                        detect_flow = "finish_top"  

                case "finish_top" :
                    if is_top(current_landmark, PoseLandMark, step["follow_through"]) :
                        step["finish_top"] = frame
                        cv2.imwrite(output_path_step, image)
                    else :
                        detect_flow = "finish"

                case "finish" :
                    if is_finish(current_landmark, PoseLandMark, step["finish_top"]) :
                        step["finish"] = frame
                        cv2.imwrite(output_path_step, image)
                        task_results[task_id] = {"status": "step_completed", "step": step}
                        print(task_results[task_id])
                        return

            if detect_flow != "address":
                if is_address(current_landmark, PoseLandMark):
                    step = {}
                    step["address"] = frame
                    is_swing=True
                    cv2.imwrite(os.path.join(image_dir+"/step", f"address.png"), image)
                    detect_flow = "take_away"

            frame += 1
        cap.release()

        if "finish_top" in step and "finish" not in step:
            step["finish"] = step["finish_top"]
            
        task_results[task_id] = {"status": "step_completed", "step": step}
        print(task_results[task_id])

    except Exception as e:
        # 작업 실패 시 오류 저장
        task_results[task_id] = {"status": "error", "error": str(e)}

@router.post("/pose_local")
async def pose_local(request: video_info, background_tasks: BackgroundTasks, app: Request): # mediapipe background
    """
    pose 추정 데이터 생성
    Args:
        request ``video_info``: 받은 요청 데이터 (비디오 경로 및 손 타입)
    Returns:
    """

    # request, config
    video_path, hand_type = request.url, request.handType
    task_results = app.app.state.task_results
    task_id = video_path
    task_results[task_id] = {"status": "processing"}

    # 백그라운드 작업 추가
    background_tasks.add_task(mp_background, task_id, video_path, request.handType, task_results)
    return {"task_id": task_id, "message": "Processing started"}
