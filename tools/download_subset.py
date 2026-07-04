#!/usr/bin/env python3

"""
Download the DICOM files corresponding to a sampled RSNA metadata subset.

Example
-------
python tools/download_subset.py \
    --input data/metadata/train_density_subset.csv \
    --output data/raw/rsna
"""

from pathlib import Path
import argparse
import subprocess
import zipfile

import pandas as pd
from tqdm import tqdm


class RSNADownloader:

    def __init__(
        self,
        input_csv: str,
        output_dir: str,
        competition: str = "rsna-breast-cancer-detection",
    ):

        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.competition = competition

        self.df = None

    ############################################################

    def load(self):

        self.df = pd.read_csv(self.input_csv)

        print(f"Loaded {len(self.df):,} images")
        print(f"Patients: {self.df.patient_id.nunique():,}")

    ############################################################

    def download(self):

        downloaded = 0
        skipped = 0
        failed = 0

        failed_files = []

        for row in tqdm(
            self.df.itertuples(index=False),
            total=len(self.df),
            desc="Downloading DICOMs",
        ):

            patient_dir = (
                self.output_dir
                / "train_images"
                / str(row.patient_id)
            )

            patient_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            dcm_file = patient_dir / f"{row.image_id}.dcm"

            if dcm_file.exists():

                skipped += 1
                continue

            remote_file = (
                f"train_images/{row.patient_id}/{row.image_id}.dcm"
            )

            command = [
                "kaggle",
                "competitions",
                "download",
                "-c",
                self.competition,
                "-f",
                remote_file,
                "-p",
                str(patient_dir),
                "--force",
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:

                failed += 1
                failed_files.append(remote_file)

                print(f"\nFailed downloading: {remote_file}")
                print(result.stderr)

                continue

            zip_file = patient_dir / f"{row.image_id}.dcm.zip"

            try:

                with zipfile.ZipFile(zip_file, "r") as z:
                    z.extractall(patient_dir)

                zip_file.unlink()

                downloaded += 1

            except Exception as e:

                failed += 1
                failed_files.append(remote_file)

                print(f"\nFailed extracting: {zip_file}")
                print(e)

        print("\n====================================")
        print("DOWNLOAD SUMMARY")
        print("====================================")

        print(f"Downloaded : {downloaded}")
        print(f"Skipped    : {skipped}")
        print(f"Failed     : {failed}")

        if failed_files:

            failed_path = (
                self.output_dir
                / "failed_downloads.txt"
            )

            with open(failed_path, "w") as f:
                f.write("\n".join(failed_files))

            print(f"\nSaved failed downloads to:")
            print(failed_path)

    ############################################################

    def run(self):

        self.load()

        self.download()


############################################################


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="CSV containing patient_id and image_id",
    )

    parser.add_argument(
        "--output",
        default="data/raw/rsna",
        help="Download directory",
    )

    args = parser.parse_args()

    downloader = RSNADownloader(
        input_csv=args.input,
        output_dir=args.output,
    )

    downloader.run()


if __name__ == "__main__":
    main()