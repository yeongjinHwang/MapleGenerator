from utils.image import load_image, visualize_parts, visualize_combined_parts
from utils.model import load_model, runs_image

from ultralytics import YOLO
import yaml

image_path = "./data/" + "bts_jhin.jpg"
image = load_image(image_path) 

# 모델 로드
hair_model = YOLO("runs/segment/train/weights/best.pt")
clothes_model = YOLO("runs/detect/train/weights/best.pt")

# Hair Segmentation 결과 추론
hair_results = runs_image(hair_model, image)
print("Hair Segmentation 결과:", hair_results)

# Clothes Detection 결과 추론
clothes_results = runs_image(clothes_model, image)
print("Clothes Detection 결과:", clothes_results)

# 두 결과를 합쳐 시각화
visualize_combined_parts(image, hair_results, clothes_results)