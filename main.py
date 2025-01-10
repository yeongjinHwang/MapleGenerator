import torch
import json
from torchvision.models import resnet18
from utils.combine import combine_segments, visualize_segments, process_hair_predictions
from utils.model import hair_loader, hair_infer, clothes_loader, clothes_infer
from utils.post_processing import preprocess_segmentation
from utils.score import extract_features, precompute_icon_features, compare_features_with_precomputed_icons, visualize_top_matches, fetch_icon

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
    "dress": "overall",
    "hair": "hair"
}

def load_item_ids(json_path):
    """
    JSON 파일에서 아이템 ID를 로드합니다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_model = resnet18(pretrained=True).eval().to(device)

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
    preprocessed_images = preprocess_segmentation(reclassified_cropped_images)
    visualize_segments(preprocessed_images, "Preprocessed Result")

    # JSON 파일 로드
    print("[INFO] JSON 데이터 로드 중...")
    hair_item_ids = load_item_ids("./ids_json/hair_item_ids.json")
    top_item_ids = load_item_ids("./ids_json/top_item_ids.json")
    overall_item_ids = load_item_ids("./ids_json/overall_item_ids.json")
    bottom_item_ids = load_item_ids("./ids_json/bottom_item_ids.json")

    # 아이콘 특징 미리 계산
    print("[INFO] 아이콘 특징 미리 계산 중...")
    hair_icon_features = precompute_icon_features(hair_item_ids, feature_model, device)
    top_icon_features = precompute_icon_features(top_item_ids, feature_model, device)
    overall_icon_features = precompute_icon_features(overall_item_ids, feature_model, device)
    bottom_icon_features = precompute_icon_features(bottom_item_ids, feature_model, device)

    # 헤어 결과 비교 및 시각화
    print("\n[RESULT] 헤어 상위 5개 아이템:")
    for class_name, seg_image in preprocessed_images.items():
        if class_name == "hair":
            seg_features = extract_features(seg_image, feature_model, device)
            top_matches = compare_features_with_precomputed_icons(seg_features, hair_icon_features, top_k=5)
            visualize_top_matches(seg_image, top_matches, hair_icon_features, overall_title="Hair Top 5 Matches")

    # 의류 결과 비교 및 시각화
    print("\n[RESULT] 의류 상위 5개 아이템:")
    for class_name, seg_image in preprocessed_images.items():
        if class_name in ["top", "overall", "bottom", "shoes"]:
            seg_features = extract_features(seg_image, feature_model, device)
            icon_features = locals()[f"{class_name}_icon_features"]
            top_matches = compare_features_with_precomputed_icons(seg_features, icon_features, top_k=5)
            visualize_top_matches(seg_image, top_matches, icon_features, overall_title=f"{class_name.capitalize()} Top 5 Matches")

if __name__ == "__main__":
    main()
