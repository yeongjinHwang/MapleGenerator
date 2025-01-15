import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.loader import fetch_icon

def visualize_segments(cropped_images,title):
    """
    재분류된 자른 세그먼트를 시각화합니다.
    Args:
        cropped_images: 재분류된 클래스별로 자른 이미지를 포함하는 딕셔너리.
    """
    num_classes = len(cropped_images)
    if num_classes == 0:
        print("시각화할 세그먼트가 없습니다.")
        return

    plt.figure(figsize=(10, 5))
    for i, (class_name, cropped_image) in enumerate(cropped_images.items()):
        plt.subplot(1, num_classes, i + 1)
        plt.imshow(cropped_image)
        plt.title(class_name)
        plt.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def process_hair_predictions(seg_results):
    """
    예측 결과를 combine_segments와 통합 가능한 포맷으로 변환합니다.
    Args:
        seg_results: 세그멘테이션 결과 리스트.
    Returns:
        combine_segments와 통합 가능한 포맷의 예측 리스트.
    """
    processed_predictions = []
    for result in seg_results:
        if "mask" in result:
            # 마스크에서 컨투어 추출
            contours, _ = cv2.findContours((result["mask"] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) > 2:  # 컨투어가 유효한 경우
                    # 포맷 변환: [(x, y), ...] -> [{"x": x, "y": y}, ...]
                    points = [{"x": int(pt[0]), "y": int(pt[1])} for pt in contour[:, 0, :]]
                    processed_predictions.append({
                        "class": result["label"],
                        "points": points
                    })
    return processed_predictions

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