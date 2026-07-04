#!/usr/bin/env python3
"""
Prepare the RSNA Breast Cancer Detection metadata.

This script:
1. Loads the original train.csv
2. Removes patients without all four mammography views
3. Saves:
    - train_all.csv
    - train_density.csv
4. Prints dataset statistics
"""

from pathlib import Path
import argparse
import json

import pandas as pd


REQUIRED_COLUMNS = [
    "patient_id",
    "image_id",
    "laterality",
    "view",
    "cancer",
    "density",
]

REQUIRED_VIEWS = {
    "LCC",
    "LMLO",
    "RCC",
    "RMLO",
}


class RSNAPreparer:

    def __init__(self, csv_path: str, output_dir: str):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = None

    ##############################################################

    def load(self):

        print(f"Loading metadata from {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

    ##############################################################

    def validate(self):

        missing = set(REQUIRED_COLUMNS) - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    ##############################################################

    def filter_complete_patients(self):

        patient_views = (
            self.df
            .groupby("patient_id")["view"]
            .apply(set)
        )

        complete_patients = patient_views[
            patient_views.apply(
                lambda x: REQUIRED_VIEWS.issubset(x)
            )
        ].index

        self.df = self.df[
            self.df.patient_id.isin(complete_patients)
        ].copy()

    ##############################################################

    def save(self):

        train_all = self.df.copy()

        train_density = self.df[
            self.df["density"].notna()
        ].copy()

        train_all_path = (
            self.output_dir / "train_all.csv"
        )

        train_density_path = (
            self.output_dir / "train_density.csv"
        )

        train_all.to_csv(
            train_all_path,
            index=False
        )

        train_density.to_csv(
            train_density_path,
            index=False
        )

        stats = {

            "patients": int(
                train_all.patient_id.nunique()
            ),

            "rows": int(
                len(train_all)
            ),

            "density_patients": int(
                train_density.patient_id.nunique()
            ),

            "density_rows": int(
                len(train_density)
            ),

            "positive_patients": int(
                train_all
                .groupby("patient_id")["cancer"]
                .max()
                .sum()
            )
        }

        with open(
            self.output_dir / "dataset_stats.json",
            "w"
        ) as f:

            json.dump(
                stats,
                f,
                indent=4
            )

        print("\n==============================")
        print("RSNA DATASET SUMMARY")
        print("==============================")

        print(f"Rows                 : {len(train_all)}")
        print(f"Patients             : {train_all.patient_id.nunique()}")
        print(f"Cancer Positive      : {stats['positive_patients']}")
        print(f"Rows w/ Density      : {len(train_density)}")
        print(f"Patients w/ Density  : {train_density.patient_id.nunique()}")

        print("\nDensity Distribution")

        print(
            train_density["density"]
            .value_counts(dropna=False)
        )

        print("\nCancer Distribution")

        print(
            train_all["cancer"]
            .value_counts()
        )

        print("\nSaved")

        print(train_all_path)
        print(train_density_path)

    ##############################################################

    def run(self):

        self.load()

        self.validate()

        self.filter_complete_patients()

        self.save()


##############################################################


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to RSNA train.csv"
    )

    parser.add_argument(
        "--output",
        default="data/metadata",
        help="Output directory"
    )

    args = parser.parse_args()

    RSNAPreparer(
        args.csv,
        args.output
    ).run()


if __name__ == "__main__":
    main()