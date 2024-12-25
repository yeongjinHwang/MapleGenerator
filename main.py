import json
from utils.combine import combine_segments, visualize_segments, process_hair_predictions
from utils.model import hair_loader, hair_infer, clothes_loader, clothes_infer
import cv2
import matplotlib.pyplot as plt
import numpy as np


# 클래스 매핑 정의
CLASS_MAPPING = {
    "bag": "top",
    "coat": "top",
    "jacket": "top",
    "shirt": "top",
    "t-shirt": "top",
    "pants": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "shoes": "shoes",
    "dress": "suit",
    "hair" : "hair"
}

def main():
    # 이미지 경로
    image_path = "./data/bts_jhin.jpg"

    # 헤어 세그멘테이션
    hair_model = hair_loader()
    hair_results = hair_infer(hair_model, image_path)
    hair_predictions = process_hair_predictions(hair_results)

    # 의류 세그멘테이션
    clothes_client = clothes_loader()
    clothes_result = clothes_infer(clothes_client, image_path)
    if isinstance(clothes_result, str):
        clothes_result = json.loads(clothes_result)
    predictions = clothes_result.get("predictions", [])

    # 헤어 결과와 의류 결과 병합
    all_predictions = predictions + hair_predictions

    # 재분류 및 세그먼트 결합
    reclassified_cropped_images = combine_segments(image_path, all_predictions, CLASS_MAPPING)

    # 시각화
    visualize_segments(reclassified_cropped_images)

    print(reclassified_cropped_images)

if __name__ == "__main__":
    main()