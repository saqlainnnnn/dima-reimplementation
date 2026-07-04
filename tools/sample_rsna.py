#!/usr/bin/env python3

"""
Sample a smaller RSNA metadata subset for rapid experimentation.

Example
-------
python tools/sample_rsna.py \
    --input data/metadata/train_density.csv \
    --output data/metadata/train_density_subset.csv \
    --patients 500
"""

from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


class RSNASampler:

    def __init__(
        self,
        input_csv: str,
        output_csv: str,
        n_patients: int,
        random_state: int = 42,
    ):

        self.input_csv = Path(input_csv)
        self.output_csv = Path(output_csv)

        self.n_patients = n_patients
        self.random_state = random_state

        self.df = None

    ###########################################################

    def load(self):

        self.df = pd.read_csv(self.input_csv)

    ###########################################################

    def sample(self):

        patient_df = (
            self.df
            .groupby("patient_id")
            .agg(
                cancer=("cancer", "max"),
                density=("density", "first"),
            )
            .reset_index()
        )

        # Create joint stratification label
        patient_df["stratum"] = (
            patient_df["cancer"].astype(str)
            + "_"
            + patient_df["density"].astype(str)
        )

        # Remove strata with fewer than 2 patients
        counts = patient_df["stratum"].value_counts()

        valid = counts[counts >= 2].index

        patient_df = patient_df[
            patient_df["stratum"].isin(valid)
        ].copy()

        if self.n_patients >= len(patient_df):

            sampled_patients = patient_df

        else:

            sampled_patients, _ = train_test_split(
                patient_df,
                train_size=self.n_patients,
                stratify=patient_df["stratum"],
                random_state=self.random_state,
            )

        sampled_ids = sampled_patients["patient_id"].tolist()

        self.df = self.df[
            self.df["patient_id"].isin(sampled_ids)
        ].copy()
        ###########################################################

    def statistics(self):

        patient_df = (
            self.df
            .groupby("patient_id")
            .agg(
                cancer=("cancer", "max"),
                density=("density", "first"),
            )
            .reset_index()
        )

        print("\n==============================")
        print("RSNA SAMPLE")
        print("==============================")

        print(f"Patients : {len(patient_df)}")
        print(f"Rows     : {len(self.df)}")

        print("\nCancer")

        print(
            patient_df["cancer"]
            .value_counts()
            .sort_index()
        )

        print("\nDensity")

        print(
            patient_df["density"]
            .value_counts()
        )

    ###########################################################

    def save(self):

        self.output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.df.to_csv(
            self.output_csv,
            index=False,
        )

        print(f"\nSaved to:\n{self.output_csv}")

    ###########################################################

    def run(self):

        self.load()

        self.sample()

        self.statistics()

        self.save()


###############################################################


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Filtered metadata CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV",
    )

    parser.add_argument(
        "--patients",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    RSNASampler(
        input_csv=args.input,
        output_csv=args.output,
        n_patients=args.patients,
        random_state=args.seed,
    ).run()


if __name__ == "__main__":
    main()