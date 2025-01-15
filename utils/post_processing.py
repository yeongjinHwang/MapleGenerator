import cv2

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

