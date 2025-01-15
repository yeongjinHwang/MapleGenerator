import cv2
import numpy as np
from collections import defaultdict

def apply_edge_detection(image):
    """
    이미지를 입력받아 Canny Edge Detection을 적용합니다.

    Args:
        image (numpy.ndarray): 입력 이미지 (RGB 또는 Grayscale).

    Returns:
        numpy.ndarray: Edge Detection 결과 이미지.
    """
    # 이미지를 Grayscale로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Canny Edge Detection 적용
    edges = cv2.Canny(gray_image, 50, 150)  # 임계값 조정 가능
    return edges

def apply_filter(image):
    """
    이미지를 입력받아 GaussianBlur를 적용합니다.

    Args:
        image (numpy.ndarray): 입력 이미지 (RGB 또는 Grayscale).

    Returns:
        numpy.ndarray: 필터링된 이미지.
    """
    # GaussianBlur 적용
    filtered_image = cv2.GaussianBlur(image, (5, 5), 0)  # 커널 크기 조정 가능
    return filtered_image

def preprocess_segmentation(cropped_images):
    """
    Segmentation 결과 이미지를 필터링 및 Edge Detection 적용.

    Args:
        cropped_images (dict): 재분류된 클래스별로 자른 이미지를 포함하는 딕셔너리.

    Returns:
        dict: 전처리된 이미지 딕셔너리.
    """
    preprocessed_images = {}
    for class_name, cropped_image in cropped_images.items():
        # Gaussian Blur 적용
        filtered_image = apply_filter(cropped_image)
        # Edge Detection 적용
        edges = apply_edge_detection(filtered_image)
        preprocessed_images[class_name] = edges
    return preprocessed_images

def combine_segments(image_path, predictions, class_mapping):
    """
    동일한 클래스로 재분류된 세그먼트를 결합하고 이미지를 자릅니다.
    Args:
        image_path: 입력 이미지의 경로.
        predictions: 세그먼트 예측 목록.
        class_mapping: 원래 클래스와 재분류된 클래스를 매핑하는 딕셔너리.
    Returns:
        재분류된 클래스별로 결합 및 자른 이미지를 포함하는 딕셔너리.
    """
    # 원본 이미지 로드
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 시각화를 위해 RGB로 변환

    # 재분류된 클래스별로 점 그룹화
    reclassified_points = defaultdict(list)
    for prediction in predictions:
        original_class = prediction["class"]
        reclassified_class = class_mapping.get(original_class, original_class)

        points = prediction.get("points", [])
        if points and isinstance(points[0], dict):  # points가 딕셔너리인 경우 변환
            points = [(int(pt["x"]), int(pt["y"])) for pt in points]
        
        reclassified_points[reclassified_class].append(np.array(points, dtype=np.int32))

    # 각 재분류된 클래스에 대해 마스크를 생성하고 이미지를 자름
    combined_cropped_images = {}
    for class_name, points_list in reclassified_points.items():
        # 동일한 재분류된 클래스의 모든 점에 대해 결합된 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for points in points_list:
            cv2.fillPoly(mask, [points], 255)

        # 마스크를 적용하여 세그먼트 자르기
        cropped = cv2.bitwise_and(image, image, mask=mask)

        # 결합된 세그먼트에 대한 바운딩 박스 찾기
        x, y, w, h = cv2.boundingRect(np.vstack(points_list))
        cropped_segment = cropped[y:y + h, x:x + w]

        # 재분류된 클래스 이름을 키로 사용하여 자른 세그먼트 저장
        combined_cropped_images[class_name] = cropped_segment

    return combined_cropped_images