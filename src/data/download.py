import os
import io
import requests
from pathlib import Path
from zipfile import ZipFile
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

import gdown
from tqdm import tqdm

from utils import listdir_nohidden

def download_from_gdrive_api(
        file_id: Path | str, file_name: Path | str, credentials: dict
    ):
    """
    Downloads a file from Google Drive.

    Args:
        file_id: The ID of the file to download.
        file_name: The name of the file to save the downloaded file as.
        credentials: A dictionary containing the credentials for the Google Drive API.

    Returns:
        None
    """
    if os.path.exists(file_name):
        print(f"File {file_name} already exists.")
        return
    # Authenticate with the Google Drive API
    creds = service_account.Credentials.from_service_account_info(credentials)
    service = build("drive", "v3", credentials=creds)

    # Download the file
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")

    return os.path.abspath(file_name)

def download_file_from_gdrive_gdown(
        url: str,
        file_name: Path | str,
        overwrite: bool = False,
        postprocess: callable = None
    ):
    print(f"Downloading {url} to {file_name}")
    # check if file exists or has been unzipped
    is_file = os.path.exists(file_name)
    is_unzipped = os.path.exists(file_name.removesuffix('.zip'))
    if not overwrite and (is_file or is_unzipped):
        raise FileExistsError(f"{file_name} exists.")

    # Make parent directory
    if not os.path.isdir(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    gdown.download(url, output=str(file_name))

    if postprocess:
        try:
            postprocess(file_name)
        except Exception as e:
            os.remove(file_name)
            raise e

def download_folder_from_gdrive_gdown(
    url: str,
    file_name: Path | str,
    overwrite: bool = False,
    postprocess: callable = None
):
    print(f"Downloading {url} to {file_name}")
    # Check if folder exists
    if not overwrite and os.path.isdir(file_name) and listdir_nohidden(file_name):
        raise FileExistsError(f"{file_name} exists and is not empty.")

    # Check if folder exists as a file
    if not overwrite and os.path.exists(file_name):
        raise FileExistsError(f"{file_name} exists and is not a directory.")

    # Make parent directory
    if not os.path.isdir(file_name):
        os.makedirs(file_name)

    gdown.download_folder(url, output=str(file_name))

    if postprocess:
        try:
            postprocess(file_name)
        except Exception as e:
            os.removedirs(file_name)
            raise e

def download_file_from_dropbox(
    url: str, file_name: Path | str, postprocess: Optional[callable] = None,
    dry_run: bool = False
):
    print(f"Downloading {url} to {file_name}.")

    if dry_run:
        return

    # headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
    r = requests.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024

    with tqdm(total=total_size, unit='B', unit_scale=True) as t:
        with open(file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=block_size):
                if chunk:
                    t.update(len(chunk))
                    f.write(chunk)

    if postprocess:
        try:
            postprocess(file_name)
        except Exception as e:
            os.remove(file_name)
            raise e

def unzip_contents(path: Path | str):
    # Check if we are directly given a zip file
    if os.path.exists(path) and os.path.splitext(path)[1] == ".zip":
        print(f"Unzipping {path}")
        with ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(path))
        os.remove(path)

        unzip_contents(path.removesuffix(".zip"))

    # Otherwise, search directory for zip files
    elif os.path.isdir(path):
        print(f"Searching {path} for zip files...")
        for d in tqdm(listdir_nohidden(path), desc=f"Unzipping {os.path.basename(path)}"):
            zip_path = os.path.join(path, d)
            if zip_path.endswith(".zip"):
                print(f"Unzipping {d}")
                with ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(path)
                os.remove(zip_path)

            # Recursively unzip the contents
                unzip_contents(path.removesuffix(".zip"))
            elif os.path.isdir(zip_path):
                unzip_contents(zip_path)
