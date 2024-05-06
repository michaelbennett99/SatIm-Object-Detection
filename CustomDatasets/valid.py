from warnings import warn

from .download import download_file_from_gdrive_gdown, unzip_contents
from constants import ROOT

VALID_PATH = ROOT / "data" / "raw" / "valid"

VALID_CATEGORY_URL = "https://drive.google.com/uc?id=1Av9tHAamg2oH_JpOpqNH1Db9C5TmKfH2"
VALID_LABELS_URL = "https://drive.google.com/uc?id=1F6Xp5QLmE9vwjwzh82Gsq1jUyzV8m7ds"
VALID_IMAGES_URL = "https://drive.google.com/uc?id=1Q1j_OgNlnJG2RlzQniFSIpLjozGAJ1D_"

def main():
    for url, end in zip(
        [VALID_CATEGORY_URL, VALID_LABELS_URL, VALID_IMAGES_URL],
        ["category.json", "labels.zip", "images.zip"]
    ):
        try:
            download_file_from_gdrive_gdown(
                url, str(VALID_PATH / end), postprocess=unzip_contents
            )
        except FileExistsError as e:
            warn(f"{e}")
