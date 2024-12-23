import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    이미지를 OpenCV를 사용해 로드하는 함수.
    
    Args:
    - image_path (str): 로드할 이미지 경로.
    
    Returns:
    - 로드된 이미지 배열.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지 경로를 찾을 수 없습니다: {image_path}")
    return image

def visualize_segmented_parts(image, results):
    """
    YOLOv11의 세그멘테이션 결과를 시각화하는 함수.

    Args:
    - image: 원본 이미지 (OpenCV 배열).
    - results: YOLOv11 모델의 예측 결과 리스트.
    """
    visualized_image = image.copy()

    # Bounding Box와 마스크 시각화
    for result in results:
        # Bounding Box 시각화
        bbox = result["box"]
        label = result["label"]
        score = result["score"]
        cv2.rectangle(
            visualized_image,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color=(0, 255, 0),  # 녹색, 더 밝게 보기 위해 사용
            thickness=4,  # 두께 증가
        )
        cv2.putText(
            visualized_image,
            f"{label} {score:.2f}",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,  # 글자 크기 증가
            (0, 255, 0),  # 녹색 텍스트
            3,  # 글자 두께 증가
        )

        # 마스크 시각화 (세그멘테이션 모델 사용 시)
        if "mask" in result:
            mask = result["mask"]
            # 마스크를 컬러로 변환하여 겹치게 시각화
            colored_mask = (mask > 0.5).astype("uint8") * 255  # 이진화된 마스크
            contours, _ = cv2.findContours(
                colored_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                cv2.drawContours(
                    visualized_image,
                    [contour],
                    -1,
                    color=(255, 0, 0),  # 파란색
                    thickness=3,  # 윤곽선 두께 증가
                )

    # Matplotlib으로 결과 표시
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
