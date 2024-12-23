import yaml

def load_config(config_path="config.yaml"):
    """
    YAML 설정 파일을 로드하는 함수.
    
    Args:
    - config_path (str): 설정 파일 경로.
    
    Returns:
    - 로드된 설정 딕셔너리.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config
