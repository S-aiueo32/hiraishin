from pathlib import Path
from typing import List


def is_image(filename: Path) -> bool:
    extenstions = ['.jpg', '.jpeg', '.png', '.bmp']
    return filename.suffix.lower() in extenstions


def get_all_images(data_dir: Path) -> List[Path]:
    return list([f for f in data_dir.glob('*') if is_image(f)])
