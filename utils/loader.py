import json
from io import BytesIO
import cv2
import requests 
import numpy as np

def load_item_ids(json_path):
    """
    JSON 파일에서 아이템 ID를 로드합니다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

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
        return None