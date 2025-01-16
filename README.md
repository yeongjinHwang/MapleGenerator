# MapleGenerator

**MapleGenerator**는 Ultralytics의 YOLOv11 모델을 활용하여 사람 이미지를 세그멘테이션하고, 각 세그멘테이션 결과를 메이플스토리 치장 아이템 스타일로 변환하는 프로젝트입니다. 이 프로젝트는 캐릭터 생성 및 커스터마이징 시스템을 개발하는 데 중점을 둡니다.

## 주요 기능

1. **세그멘테이션**
   - 입력된 사람 이미지에서 '머리', '상의', '하의', '신발', '한벌옷' 영역을 Fine-Tuning YOLOv11 모델, pre-trained Model의 Hybrid Model을 통해 세그멘테이션합니다.

2. **이미지 검색**
   - MapleStory API를 통해 subCategory(Class) 5개의 항목에 대해 Item Icon을 받아옵니다.
   - Icon들을 ResNET50을 통해 Embedding하고, Embedding 결과에 대해 FARISS를 통해 Embedding Area 저장합니다.
   - 입력된 이미지와 Class기반으로 Ebedding Area에서 각각 TOP5 Icon을 추출합니다

## 사용 방법

MapleStory API를 통해 ICON들을 Load, Save:

```bash
python icon.py
```
해당 ICON 폴더의 모든 Icon Image들을 Embedding 및 결과 저장
```bash
python embedding.py
```
해당 ICON Embedding 결과들과 입력 이미지 score계산 및 TOP5 추출
```bash
python main.py
```


## 디렉토리 구조

```
MapleGenerator/
├── data/
│   ├── embeddings
│   │   ├── *_id_map.json        # Embedding Idx와 Icon Ids Mapping
│   │   └── *_index.bin          # Embedding 저장
│   ├── icon                     
│   │   └── {subCategory}/        # Icon Image 저장
├── dataset/             # 데이터셋 폴더
├── runs/                # weight file 저장 폴더
├── utils/               
│   ├── combine.py      
│   ├── model.py     
│   ├── post_processing.py 
│   └── visual.py
├── icon.py
├── embedding.py
└── main.py
```

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 문의

프로젝트 관련 문의사항은 아래 이메일로 연락해주세요:
- Email: yeongjin.gongjin@gmail.com
