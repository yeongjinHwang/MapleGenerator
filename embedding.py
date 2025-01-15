import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from utils.post_processing import edge_detection
import faiss
import json

# Pre-trained 모델 로드 (ResNet50 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device)
resnet.fc = torch.nn.Identity()  # 마지막 레이어 제거
resnet.eval()

# 이미지 전처리 함수
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class IconDataset(Dataset):
    """
    아이콘 이미지 경로와 파일명을 제공하는 PyTorch Dataset.
    """
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = transforms.ToPILImage()(image_rgb)
        input_tensor = preprocess(pil_image)
        return input_tensor, image_path

def embedding_by_category(icon_dir, faiss_index_dir, batch_size=32):
    """
    카테고리별로 아이콘 임베딩을 생성하고 FAISS 인덱스와 ID 매핑을 저장.

    Args:
        icon_dir (str): 아이콘 이미지가 저장된 루트 디렉토리.
        faiss_index_dir (str): FAISS 인덱스를 저장할 루트 디렉토리.
        batch_size (int): 배치 크기.
    """
    os.makedirs(faiss_index_dir, exist_ok=True)

    for subcategory in os.listdir(icon_dir):
        subcategory_path = os.path.join(icon_dir, subcategory)
        if not os.path.isdir(subcategory_path):
            continue

        # FAISS Index 생성
        index = faiss.IndexFlatL2(2048)  # L2 거리 기반 검색

        # 서브카테고리 내부 파일 경로 수집
        image_paths = [
            os.path.join(subcategory_path, file_name)
            for file_name in os.listdir(subcategory_path)
            if file_name.endswith(".png")
        ]

        # ID 매핑 리스트 생성
        id_mapping = []

        # PyTorch Dataset 및 DataLoader 생성
        dataset = IconDataset(image_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f"[INFO] '{subcategory}'에서 총 {len(image_paths)}개의 이미지를 처리합니다.")
        for batch in dataloader:
            input_tensors, batch_image_paths = batch
            input_tensors = input_tensors.to(device)

            # 임베딩 생성
            with torch.no_grad():
                batch_embeddings = resnet(input_tensors)

            # FAISS 인덱스에 추가
            index.add(batch_embeddings.cpu().numpy())

            # 배치 이미지 경로에서 ID 추출 및 매핑
            batch_ids = [os.path.splitext(os.path.basename(path))[0] for path in batch_image_paths]
            id_mapping.extend(batch_ids)

        # FAISS 인덱스와 ID 매핑 저장
        index_path = os.path.join(faiss_index_dir, f"{subcategory}_index.bin")
        id_map_path = os.path.join(faiss_index_dir, f"{subcategory}_id_map.json")

        faiss.write_index(index, index_path)
        with open(id_map_path, "w") as f:
            json.dump(id_mapping, f, indent=4)

        print(f"[INFO] '{subcategory}' 카테고리의 FAISS 인덱스 저장 완료: {index_path}")
        print(f"[INFO] '{subcategory}' 카테고리의 ID 매핑 저장 완료: {id_map_path}")


if __name__ == "__main__":
    # 아이콘 디렉토리와 임베딩 저장 디렉토리 경로
    icon_dir = "./data/icon"  # 아이콘 이미지가 저장된 루트 디렉토리
    embedding_dir = "./data/embeddings"  # 임베딩을 저장할 루트 디렉토리

    # GPU를 사용하여 배치 임베딩 처리
    embedding_by_category(icon_dir, embedding_dir, batch_size=32)
