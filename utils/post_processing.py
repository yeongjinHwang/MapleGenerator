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

def edge_detection(input_data):
    """
    Segmentation 결과 이미지를 필터링 및 Edge Detection 적용.
    단일 이미지 또는 클래스별 이미지 딕셔너리를 모두 처리 가능.

    Args:
        input_data (dict or numpy.ndarray): 
            - dict: 재분류된 클래스별로 자른 이미지를 포함하는 딕셔너리.
            - numpy.ndarray: 단일 이미지.

    Returns:
        dict or numpy.ndarray: 
            - dict: 전처리된 이미지 딕셔너리.
            - numpy.ndarray: 단일 이미지의 전처리 결과.
    """
    def process_image(image):
        # Gaussian Blur 적용
        filtered_image = apply_filter(image)
        # Edge Detection 적용
        edges = apply_edge_detection(filtered_image)
        return edges

    if isinstance(input_data, dict):  # 입력 데이터가 dict인 경우
        preprocessed_images = {}
        for class_name, cropped_image in input_data.items():
            preprocessed_images[class_name] = process_image(cropped_image)
        return preprocessed_images

    elif isinstance(input_data, np.ndarray):  # 입력 데이터가 단일 이미지인 경우
        return process_image(input_data)

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