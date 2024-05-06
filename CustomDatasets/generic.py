import argparse
from warnings import warn

from .download import download_folder_from_gdrive_gdown, unzip_contents

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download data from Google Drive"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL to the Google Drive folder",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to save the downloaded data",
    )
    return parser.parse_args()

def main(args):
    try:
        download_folder_from_gdrive_gdown(
            args.url, args.path, postprocess=unzip_contents
        )
    except FileExistsError as e:
        warn(f"{e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
