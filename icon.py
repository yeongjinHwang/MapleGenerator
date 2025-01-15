import requests
import os
import cv2
import numpy as np
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_and_save_icon(item_id, subcategory, output_dir, region="KMS", version="389"):
    """
    maplestory.io API를 통해 아이템 ID에 해당하는 아이콘을 가져와 저장하는 함수.

    Args:
        item_id (int): 아이템 ID.
        subcategory (str): 서브카테고리 이름 (예: "Shoes").
        output_dir (str): 저장할 디렉토리의 기본 경로.
        region (str): 지역 코드.
        version (str): 버전 정보.

    Returns:
        bool: 성공 여부.
    """
    # API URL
    url = f"https://maplestory.io/api/{region}/{version}/item/{item_id}/icon"
    response = requests.get(url)
    
    if response.status_code == 200:
        # 아이콘 데이터를 OpenCV 형식으로 변환
        image_data = BytesIO(response.content).read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # 저장 디렉토리 생성
        category_dir = os.path.join(output_dir, subcategory)
        os.makedirs(category_dir, exist_ok=True)

        # 파일 경로 설정
        file_path = os.path.join(category_dir, f"{item_id}.png")
        cv2.imwrite(file_path, image)  # 이미지 저장
        return True
    else:
        return False

def process_subcategory(subcategory, item_ids, output_dir, region, version):
    """
    특정 서브카테고리를 처리하여 아이콘을 저장하는 함수.

    Args:
        subcategory (str): 서브카테고리 이름.
        item_ids (list): 아이템 ID 리스트.
        output_dir (str): 저장할 디렉토리의 기본 경로.
        region (str): 지역 코드.
        version (str): 버전 정보.
    """
    print(f"[INFO] '{subcategory}'에서 총 {len(item_ids)}개의 아이템을 처리합니다.")
    success_count = 0

    # 병렬 처리
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_id = {
            executor.submit(fetch_and_save_icon, item_id, subcategory, output_dir, region, version): item_id
            for item_id in item_ids
        }

        for future in as_completed(future_to_id):
            item_id = future_to_id[future]
            success = future.result()
            if success:
                success_count += 1

    print(f"[INFO] '{subcategory}' 처리 완료 - 총 {success_count}/{len(item_ids)}개의 아이콘이 저장되었습니다.")

def fetch_item_ids_by_category(region, version, subcategories, output_dir):
    """
    전체 아이템 데이터에서 특정 서브카테고리의 아이템을 필터링하고, 각 아이콘을 저장하는 함수.

    Args:
        region (str): 지역 코드.
        version (str): 버전 정보.
        subcategories (dict): 서브카테고리와 아이템 ID 리스트를 매핑한 딕셔너리.
        output_dir (str): 저장할 디렉토리의 기본 경로.
    """
    try:
        # 전체 아이템 목록 API URL
        url = f"https://maplestory.io/api/{region}/{version}/item"
        response = requests.get(url)

        if response.status_code == 200:
            items = response.json()

            for subcategory, _ in subcategories.items():
                # 서브카테고리에 해당하는 아이템 필터링
                item_ids = [
                    item['id'] for item in items
                    if item.get('typeInfo', {}).get('subCategory') == subcategory
                ]

                # 서브카테고리 처리
                process_subcategory(subcategory, item_ids, output_dir, region, version)
        else:
            print(f"[ERROR] 전체 아이템 API 요청 실패: 상태 코드 {response.status_code}")
    except Exception as e:
        print(f"[ERROR] 아이템 처리 중 오류 발생: {e}")


# 실행 예시
if __name__ == "__main__":
    region = "KMS"  # 지역 코드
    version = "389"  # 버전 정보

    # 저장할 서브카테고리 이름
    subcategories = {
        "Shoes": None,
        "Hair": None,
        "Top": None,
        "Overall": None,
        "Bottom": None
    }

    # 아이콘 저장 디렉토리 설정
    output_dir = "./data/icon"

    # 함수 호출
    fetch_item_ids_by_category(region, version, subcategories, output_dir)
