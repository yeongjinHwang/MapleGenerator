from ultralytics import YOLO
import cv2

def load_model(model_path="yolo11m-seg.pt"):
    """
    YOLOv11 모델을 로드하는 함수.
    
    Args:
    - model_path (str): YOLOv11 모델 파일 경로.
    
    Returns:
    - 로드된 모델 객체.
    """
    return YOLO(model_path)

def segment_image(model, image):
    """
    YOLOv11 모델을 사용해 세그멘테이션 및 박스 예측을 수행하는 함수.

    Args:
    - model: 로드된 YOLOv11 모델 객체.
    - image: 입력 이미지 (OpenCV 배열).

    Returns:
    - 예측 결과 리스트 (박스, 라벨, 점수, 마스크 포함).
    """
    # YOLO 예측 수행
    results = model.predict(image, save=False, save_txt=False)

    # 원본 이미지 크기
    orig_height, orig_width = image.shape[:2]

    # YOLO 모델이 처리한 입력 이미지 크기
    yolo_height, yolo_width = results[0].orig_img.shape[:2]

    predictions = []

    for i, box in enumerate(results[0].boxes):
        # Bounding Box 좌표 변환
        bbox = box.xyxy[0].cpu().numpy()  # 리사이즈된 좌표
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * (orig_width / yolo_width))
        y1 = int(y1 * (orig_height / yolo_height))
        x2 = int(x2 * (orig_width / yolo_width))
        y2 = int(y2 * (orig_height / yolo_height))

        prediction = {
            "box": [x1, y1, x2, y2],  # 원본 이미지 크기로 변환된 박스
            "label": results[0].names[int(box.cls[0].item())],  # 클래스 이름
            "score": box.conf[0].item(),  # 신뢰도
        }

        # 마스크 변환 (세그멘테이션 모델 사용 시)
        if results[0].masks is not None:
            mask = results[0].masks.data[i].cpu().numpy()  # 리사이즈된 마스크
            resized_mask = cv2.resize(mask, (orig_width, orig_height))  # 원본 크기로 변환
            prediction["mask"] = resized_mask

        predictions.append(prediction)

    return predictions
