import requests
import json

def fetch_item_ids_by_category(region, version, subcategories):
    """
    전체 아이템 데이터에서 특정 서브카테고리의 아이템을 필터링하고, 각 서브카테고리의 아이템 ID를 JSON으로 저장하는 함수.

    Args:
        region (str): 지역 코드 (예: "KMS").
        version (str): 버전 정보 (예: "latest").
        subcategories (dict): 서브카테고리와 파일명을 매핑한 딕셔너리.
    """
    try:
        # 전체 아이템 목록 API URL
        url = f"https://maplestory.io/api/{region}/{version}/item"
        response = requests.get(url)

        # 응답 데이터 확인
        if response.status_code == 200:
            items = response.json()

            # 각 서브카테고리별로 ID 추출 및 저장
            for subcategory, filename in subcategories.items():
                filename = './ids_json/'+filename
                filtered_ids = [
                    item['id'] for item in items 
                    if item.get('typeInfo', {}).get('subCategory') == subcategory
                ]
                print(f"총 {len(filtered_ids)}개의 '{subcategory}' 아이템을 발견했습니다.")

                # JSON 파일로 저장
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(filtered_ids, f, ensure_ascii=False, indent=4)

                print(f"'{subcategory}' 아이템 ID가 '{filename}'에 저장되었습니다.")
        else:
            print(f"전체 아이템 API 요청 실패: 상태 코드 {response.status_code}")
    except Exception as e:
        print(f"오류 발생: {e}")

# 사용 예시
region = "KMS"  # 지역 코드
version = "389"  # 버전 정보

# 저장할 서브카테고리와 파일명 매핑
subcategories = {
    "Hair": "hair_item_ids.json",
    "Top": "top_item_ids.json",
    "Overall": "overall_item_ids.json",
    "Bottom": "bottom_item_ids.json"
}

# 함수 호출
fetch_item_ids_by_category(region, version, subcategories)
