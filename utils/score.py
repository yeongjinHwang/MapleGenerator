import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from io import BytesIO

def fetch_icon(item_id, region="KMS", version="389"):
    """
    maplestory.io API를 통해 아이템 ID에 해당하는 아이콘 이미지를 가져오는 함수.

    Args:
        item_id (int): 아이템 ID.
        region (str): 지역 코드 (기본값: "KMS").
        version (str): 버전 정보 (기본값: "389").

    Returns:
        numpy.ndarray or None: 가져온 이미지 객체 (BGR 형식) 또는 실패 시 None.
    """
    url = f"https://maplestory.io/api/{region}/{version}/item/{item_id}/icon"
    response = requests.get(url)
    if response.status_code == 200:
        # API 응답 데이터를 OpenCV 형식으로 변환
        image_data = BytesIO(response.content).read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        return image
    else:
        print(f"아이콘 불러오기 실패: ID {item_id}, 상태 코드 {response.status_code}")
        return None


def extract_features(image, model, device):
    """
    이미지를 입력받아 특징 벡터를 추출합니다.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()

def precompute_icon_features(item_ids, feature_model, device):
    """
    아이콘 ID에 해당하는 특징 벡터를 미리 계산합니다.
    """
    icon_features = {}
    for item_id in item_ids:
        icon_image = fetch_icon(item_id)
        if icon_image is not None:
            features = extract_features(icon_image, feature_model, device)
            icon_features[item_id] = features
    return icon_features

def compare_features_with_precomputed_icons(seg_features, icon_features, top_k=5):
    """
    Segmentation 특징과 미리 계산된 아이콘 특징 벡터를 비교합니다.
    """
    similarities = []
    for item_id, features in icon_features.items():
        similarity = cosine_similarity(seg_features, features)[0][0]
        similarities.append((item_id, similarity))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

def visualize_top_matches(seg_image, top_matches, icon_features, overall_title="Top Matches"):
    """
    Segmentation 이미지와 상위 5개의 아이템 이미지를 시각화하고 유사도 점수를 표시합니다.

    Args:
        seg_image (numpy.ndarray): Segmentation 결과 이미지.
        top_matches (list): [(item_id, 유사도)] 형태의 상위 아이템 리스트.
        icon_features (dict): {item_id: 이미지 데이터(numpy.ndarray)} 딕셔너리.
        overall_title (str): 그래프 전체 제목 (기본값: "Top Matches").
    """
    num_matches = len(top_matches)
    total_columns = num_matches + 1  # Segmentation 이미지 + 상위 5개 아이템

    plt.figure(figsize=(15, 5))

    # 첫 번째 컬럼: Segmentation 이미지
    plt.subplot(1, total_columns, 1)
    plt.imshow(cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB))  # BGR -> RGB 변환
    plt.title("Segmentation\nImage", fontsize=10)
    plt.axis("off")

    # 나머지 컬럼: 상위 매칭 결과
    for i, (item_id, similarity) in enumerate(top_matches):
        icon_image = fetch_icon(item_id)
        if icon_image is None:
            print(f"ID {item_id}의 아이콘 이미지를 불러올 수 없습니다.")
            continue

        plt.subplot(1, total_columns, i + 2)  # Segmentation 이미지 이후부터 시작
        plt.imshow(cv2.cvtColor(icon_image, cv2.COLOR_BGR2RGB))  # BGR -> RGB 변환
        plt.title(f"ID: {item_id}\nScore: {similarity:.2f}", fontsize=10)
        plt.axis("off")

    # 전체 제목 추가
    plt.suptitle(overall_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 제목과 겹치지 않도록 조정
    plt.show()