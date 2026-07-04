#!/usr/bin/env python3

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


class RSNAPreparer:

    def __init__(self, csv_path: str, output_dir: str):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = None

    ##########################################################

    def load(self):

        print(f"Loading {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        print(f"Rows: {len(self.df):,}")
        print(f"Patients: {self.df.patient_id.nunique():,}")

    ##########################################################

    def validate(self):

        missing = set(REQUIRED_COLUMNS) - set(self.df.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

    ##########################################################

    def remove_duplicates(self):

        before = len(self.df)

        self.df = self.df.drop_duplicates(
            subset=["patient_id", "image_id"]
        )

        after = len(self.df)

        print(f"Removed {before-after} duplicate rows")

    ##########################################################

    def filter_complete_patients(self):

        REQUIRED = {
            ("L", "CC"),
            ("L", "MLO"),
            ("R", "CC"),
            ("R", "MLO"),
        }

        patient_views = (
            self.df
            .groupby("patient_id")
            .apply(
                lambda x: set(
                    zip(
                        x["laterality"],
                        x["view"]
                    )
                ),
                include_groups=False
            )
        )

        complete_patients = patient_views[
            patient_views.apply(
                lambda x: REQUIRED.issubset(x)
            )
        ].index

        print(
            f"Patients with all four views: {len(complete_patients):,}"
        )

        self.df = self.df[
            self.df.patient_id.isin(
                complete_patients
            )
        ].copy()

    ##########################################################

    def save(self):

        train_all = self.df.copy()

        train_density = self.df[
            self.df["density"].notna()
        ].copy()

        density_per_patient = (
            train_density
            .groupby("patient_id")["density"]
            .nunique()
        )

        assert (density_per_patient == 1).all(), (
            "Some patients have multiple density labels!"
        )

        train_all.to_csv(
            self.output_dir / "train_all.csv",
            index=False
        )

        train_density.to_csv(
            self.output_dir / "train_density.csv",
            index=False
        )

        stats = {

            "patients":
                int(train_all.patient_id.nunique()),

            "rows":
                int(len(train_all)),

            "density_patients":
                int(train_density.patient_id.nunique()),

            "density_rows":
                int(len(train_density)),

            "positive_patients":
                int(
                    train_all
                    .groupby("patient_id")["cancer"]
                    .max()
                    .sum()
                ),

            "density_distribution":
                train_density["density"]
                .value_counts(dropna=False)
                .to_dict(),

            "cancer_distribution":
                train_all["cancer"]
                .value_counts()
                .to_dict(),
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

        print(f"Rows                 : {stats['rows']:,}")
        print(f"Patients             : {stats['patients']:,}")
        print(f"Positive Patients    : {stats['positive_patients']:,}")
        print(f"Rows with Density    : {stats['density_rows']:,}")
        print(f"Patients with Density: {stats['density_patients']:,}")

        print("\nDensity Distribution")

        print(train_density["density"].value_counts(dropna=False))

        print("\nCancer Distribution")

        print(train_all["cancer"].value_counts())

        print("\nSaved")

        print(self.output_dir / "train_all.csv")
        print(self.output_dir / "train_density.csv")
        print(self.output_dir / "dataset_stats.json")

    ##########################################################

    def run(self):

        self.load()

        self.validate()

        self.remove_duplicates()

        self.filter_complete_patients()

        self.save()


##############################################################


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True
    )

    parser.add_argument(
        "--output",
        default="data/metadata"
    )

    args = parser.parse_args()

    RSNAPreparer(
        csv_path=args.csv,
        output_dir=args.output
    ).run()


if __name__ == "__main__":
    main()