import faiss
import cv2
import json
import os
import torch
from torchvision import models, transforms
from utils.combine import combine_segments
from utils.model import hair_loader, hair_infer, clothes_loader, clothes_infer
from utils.post_processing import edge_detection, combine_segments, process_hair_predictions
from utils.visual import visualize_segments, visualize_top_matches
# 클래스 매핑 정의
CLASS_MAPPING = {
    "bag": "None",
    "coat": "Overall",
    "dress": "Overall",
    "jacket": "Top",
    "shirt": "Top",
    "t-shirt": "Top",
    "pants": "Bottom",
    "shorts": "Bottom",
    "skirt": "Bottom",
    "shoes": "Shoes",
    "hair": "Hair"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device)
resnet.fc = torch.nn.Identity()
resnet.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_embedding(image, model, preprocess_fn):
    """
    이미지에서 임베딩 생성
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = transforms.ToPILImage()(image_rgb)
    input_tensor = preprocess_fn(pil_image).unsqueeze(0).to(device)  # 배치 추가 및 GPU 이동
    with torch.no_grad():
        embedding = model(input_tensor).squeeze(0).cpu().numpy()  # 배치 제거 및 NumPy 변환
    return embedding

def search_faiss(seg_embedding, faiss_index_path, id_map_path, top_k=5):
    """
    FAISS 인덱스에서 유사도 검색 후 ID를 반환.

    Args:
        seg_embedding (np.ndarray): 검색할 세그멘테이션 임베딩 (1, 2048).
        faiss_index_path (str): FAISS 인덱스 파일 경로.
        id_map_path (str): ID 매핑 JSON 파일 경로.
        top_k (int): 검색할 상위 K개.

    Returns:
        list of tuple: 상위 K개의 결과 (ID, Distance).
    """
    if not os.path.exists(faiss_index_path) or not os.path.exists(id_map_path):
        print(f"[WARN] FAISS 인덱스 또는 ID 매핑 파일이 존재하지 않습니다.")
        return []

    # FAISS 인덱스 로드
    index = faiss.read_index(faiss_index_path)

    # ID 매핑 로드
    with open(id_map_path, "r") as f:
        id_mapping = json.load(f)

    # 검색
    D, I = index.search(seg_embedding, top_k)
    return [(id_mapping[I[0][i]], D[0][i]) for i in range(top_k)]

def main():
    faiss_index_dir = "./data/embeddings"   # FAISS 인덱스 디렉토리
    icon_dir = "./data/icon"  # 아이콘 이미지 저장 디렉토리

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
    reclassified_images = combine_segments(image_path, all_predictions, CLASS_MAPPING)
    visualize_segments(reclassified_images, "Segmentation Result")

    # 아이콘과 비교
    print("[INFO] FAISS를 사용하여 비교 중...")
    icon_features = {}
    for class_name, seg_image in reclassified_images.items():
        # 세그멘테이션 이미지의 임베딩 생성
        seg_embedding = generate_embedding(seg_image, resnet, preprocess).reshape(1, -1)  # (1, 2048)

        # FAISS 인덱스 경로
        faiss_index_path = os.path.join(faiss_index_dir, f"{class_name}_index.bin")
        id_map_path = os.path.join(faiss_index_dir, f"{class_name}_id_map.json")

        # FAISS 검색
        results = search_faiss(seg_embedding, faiss_index_path, id_map_path, top_k=5)
        icon_features[class_name] = results

        # Segmentation 이미지와 TOP 5 결과 시각화
        visualize_top_matches(
            seg_image,
            results,
            icon_dir=os.path.join(icon_dir, class_name),
            overall_title=f"Top 5 Matches for {class_name.capitalize()}"
        )
    # 유사도 결과 출력
    for class_name, results in icon_features.items():
        print(f"\n[RESULT] {class_name}:")
        for idx, distance in results:
            print(f"  - Index: {idx}, Distance: {distance:.4f}")


if __name__ == "__main__":
    main()
