# MapleGenerator

**MapleGenerator**는 Ultralytics의 YOLOv11 모델을 활용하여 사람 이미지를 세그멘테이션하고, 각 세그멘테이션 결과를 메이플스토리 치장 아이템 스타일로 변환하는 프로젝트입니다. 이 프로젝트는 캐릭터 생성 및 커스터마이징 시스템을 개발하는 데 중점을 둡니다.

## 주요 기능

1. **세그멘테이션**
   - 입력된 사람 이미지에서 '머리', '상의', '하의', '신발', '얼굴(표정)' 영역을 YOLOv11 모델을 통해 세그멘테이션합니다.

2. **이미지 변환**
   - 세그멘테이션 결과를 Image Generator를 사용하여 메이플스토리 치장 아이템 스타일로 변환합니다.

3. **캐릭터 생성**
   - 변환된 치장 아이템을 메이플스토리 기본 캐릭터에 합성하여 최종 캐릭터를 생성합니다.

## 설치 방법

### 요구 사항
- Python 3.8 이상
- PyTorch 1.10 이상
- Ultralytics YOLOv11

4. `hair_custom.yaml` 파일을 설정합니다. 프로젝트 설정은 `utils` 폴더에서 관리됩니다.

## 사용 방법

모든 작업은 `main.py`를 통해 수행됩니다. 아래의 명령어로 실행하세요:

```bash
python main.py --input <이미지 경로> --output <결과 저장 경로> # 추후 수정해야됩니다.
```

## 디렉토리 구조

```
MapleGenerator/
├── data/                # 데이터 파일 및 샘플 이미지
├── dataset/             # 데이터셋 폴더
├── ids_json/            # JSON 파일 관리
├── runs/                # 실행 결과 저장 디렉토리
├── utils/               # 유틸리티 코드 관리
│   ├── combine.py       # 세그멘테이션 결과 병합
│   ├── model.py         # 모델 로드 및 처리 함수
│   ├── post_processing.py # 후처리 관련 함수
│   ├── score.py         # 점수 계산 관련 함수
│   └── get_id_json.py   # JSON ID 생성
├── hair_custom.yaml     # 사용자 정의 설정 파일
└── main.py              # 메인 스크립트
```

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 문의

프로젝트 관련 문의사항은 아래 이메일로 연락해주세요:
- Email: yeongjin.hwang@example.com
