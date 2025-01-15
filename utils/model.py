from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
import cv2

def clothes_loader(api_url="https://outline.roboflow.com", api_key="D2rCGXVyfqnwnPrSIokV"):
    """
    의류 세그멘테이션 모델 로드 함수.
    Args:
        api_url: API URL (기본값: Roboflow API URL).
        api_key: API 키.
    Returns:
        CLIENT: InferenceHTTPClient 객체.
    """
    CLIENT = InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key
    )
    return CLIENT

def clothes_infer(client, image_path, model_id="clothing-segmentation-dataset/1"):
    """
    의류 세그멘테이션 추론 함수.
    Args:
        client: InferenceHTTPClient 객체.
        image_path: 입력 이미지 경로.
        model_id: 모델 ID (기본값: "clothing-segmentation-dataset/1").
    Returns:
        result: 추론 결과 객체.
    """
    # 추론 수행
    result = client.infer(image_path, model_id=model_id)
    return result

def hair_loader():
    """
    헤어 세그멘테이션 모델 로드 함수.
    Returns:
        hair_model: YOLO 모델 객체.
    """
    hair_model = YOLO("runs/hair_segment/train/weights/best.pt")
    return hair_model

def hair_infer(model, image_path):
    """
    YOLOv11 모델을 사용해 세그멘테이션 및 박스 예측을 수행하는 함수.
    Args:
    - model: 로드된 YOLOv11 모델 객체.
    - image_path: 입력 이미지 경로 (문자열).
    Returns:
    - predictions: 예측 결과 리스트 (박스, 라벨, 점수, 마스크 포함).
    """
    # 입력 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    orig_height, orig_width = image.shape[:2]

    # YOLO 예측 수행
    results = model.predict(image, save=False, save_txt=False)

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