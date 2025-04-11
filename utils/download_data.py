import argparse
import os

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm.auto import tqdm

ENDPOINT_URL = "https://os.unil.cloud.switch.ch"
BUCKET_NAME = "lts2-riboclette"


class ProgressBar:
    def __init__(self, filesize):
        self._progress = tqdm(
            total=filesize, unit="B", unit_scale=True, desc="Downloading"
        )

    def __call__(self, bytes_amount):
        self._progress.update(bytes_amount)


def download_files_from_s3(
    data_dir, allow_overwrite: bool = False, prefixes: str = None
):

    s3 = boto3.client(
        "s3", endpoint_url=ENDPOINT_URL, config=Config(signature_version=UNSIGNED)
    )

    list_objects_v2_kwargs = dict(Bucket=BUCKET_NAME)
    if prefixes:
        for prefix in prefixes:
            list_objects_v2_kwargs["Prefix"] = prefix
            _download_files_from_s3(
                s3, data_dir, allow_overwrite, list_objects_v2_kwargs
            )
    else:
        _download_files_from_s3(s3, data_dir, allow_overwrite, list_objects_v2_kwargs)


def _download_files_from_s3(s3, data_dir, allow_overwrite, list_objects_v2_kwargs):
    response = s3.list_objects_v2(**list_objects_v2_kwargs)
    if "Contents" in response:
        for obj in response["Contents"]:
            file_key = obj["Key"]

            print(f"Downloading {file_key}...\n")

            metadata = s3.head_object(Bucket=BUCKET_NAME, Key=file_key)
            filesize = metadata["ContentLength"]

            local_file_path = os.path.join(data_dir, *file_key.split("/"))

            if os.path.exists(local_file_path) and not allow_overwrite:
                print(f"Skipping {local_file_path} (already exists)")
                continue

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            s3.download_file(
                BUCKET_NAME, file_key, local_file_path, Callback=ProgressBar(filesize)
            )
    else:
        print("No files found with the specified prefix.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download files from an S3 bucket based on a prefix."
    )

    parser.add_argument(
        "--setting",
        choices=["all", "figures", "model"],
        required=True,
        help="Specify the type of files to download. Choose from 'all', 'figures', or 'model'.",
    )

    parser.add_argument(
        "--data_dir",
        nargs="?",
        default=os.getcwd(),
        help="The local folder where downloaded files will be saved (default: current workdir)",
    )

    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow overwriting existing files",
    )

    args = parser.parse_args()

    prefixes = ""
    if args.setting == "figures":
        prefixes = ["data/data/", "data/results"]
    if args.setting == "model":
        prefixes = [
            "checkpoints",
            "data/Lina",
            "data/Liver",
            "data/processed",
            "data/orig",
        ]

    download_files_from_s3(args.data_dir, args.allow_overwrite, prefixes=prefixes)
