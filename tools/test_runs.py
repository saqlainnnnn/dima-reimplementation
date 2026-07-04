import pandas as pd

df = pd.read_csv("data/metadata/train_density_subset.csv")

df.head(5).to_csv(
    "data/metadata/test_download.csv",
    index=False,
)