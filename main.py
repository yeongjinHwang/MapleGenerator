import torch
import json
import heapq

from utils.combine import combine_segments
from utils.model import hair_loader, hair_infer, clothes_loader, clothes_infer
from utils.post_processing import edge_segmentation, combine_segments, process_hair_predictions
from utils.visual import visualize_segments, visualize_top_matches
from utils.loader import load_item_ids, fetch_icon

# 클래스 매핑 정의
CLASS_MAPPING = {
    "bag": "none",
    "coat": "overall",
    "dress": "overall",
    "jacket": "top",
    "shirt": "top",
    "t-shirt": "top",
    "pants": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "shoes": "shoes",
    "hair": "hair"
}

def main():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 입력 이미지 경로
    image_path = "./data/bts_jhin.jpg"

    # 헤어 세그멘테이션
    print("[INFO] 헤어 세그멘테이션 모델 로드 중...")
    hair_model = hair_loader()
    hair_results = hair_infer(hair_model, image_path)
    hair_predictions = process_hair_predictions(hair_results)

    # 의류 탐지
    print("[INFO] 의류 탐지 모델 로드 중...")
    clothes_client = clothes_loader()
    clothes_result = clothes_infer(clothes_client, image_path)
    if isinstance(clothes_result, str):
        clothes_result = json.loads(clothes_result)
    predictions = clothes_result.get("predictions", [])

    # 헤어와 의류 결과 병합
    all_predictions = predictions + hair_predictions

    # 재분류 및 세그먼트 결합
    print("[INFO] 세그멘테이션 결과 병합 중...")
    reclassified_cropped_images = combine_segments(image_path, all_predictions, CLASS_MAPPING)
    visualize_segments(reclassified_cropped_images, "Segmentation Result")

    # 전처리 적용
    print("[INFO] 세그멘테이션 결과 전처리 중...")
    preprocessed_images = edge_segmentation(reclassified_cropped_images)
    visualize_segments(preprocessed_images, "Preprocessed Result")

    # JSON 파일 로드
    print("[INFO] JSON 데이터 로드 중...")
    item_ids = {
        "hair" : load_item_ids("./ids_json/hair_item_ids.json"),
        "top_item_ids" : load_item_ids("./ids_json/top_item_ids.json"),
        "overall_item_ids" : load_item_ids("./ids_json/overall_item_ids.json"),
        "bottom_item_ids" : load_item_ids("./ids_json/bottom_item_ids.json")
    }

    # 아이콘과 비교
    print("[INFO] 비교 중...")
    icon_features = {}
    for class_name, seg_image in preprocessed_images.items() : 
        top_5_heap = []  # 최소 힙으로 상위 5개를 유지
        for item_ids in item_ids[class_name]:
            icon_image = fetch_icon(item_ids)
            if icon_image is not None : 
                preprocessed_item = edge_segmentation(icon_image)
                score = 1 #임시, preprocessed_item와 seg_image score 계산

                # 힙에 (score, item_id) 추가, 5개를 초과하면 최소값과 비교 후 대체
                if len(top_5_heap) < 5: heapq.heappush(top_5_heap, (score, item_ids))
                else: heapq.heappushpop(top_5_heap, (score, item_ids))

        icon_features[class_name] = sorted(top_5_heap, key=lambda x: x[0], reverse=True)

if __name__ == "__main__":
    main()
