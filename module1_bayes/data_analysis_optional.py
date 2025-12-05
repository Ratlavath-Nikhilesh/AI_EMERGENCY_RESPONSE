# module1_bayes/data_analysis_optional.py

import os
import pandas as pd


DATA_PATH = os.path.join("data", "india_road_accident_severity.csv")


def basic_severity_stats():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    if "Accident_severity" not in df.columns:
        print("Column 'Accident_severity' not found. Available columns:")
        print(df.columns.tolist())
        return

    counts = df["Accident_severity"].value_counts(normalize=True)
    print("Empirical severity distribution in the dataset:")
    print(counts)
    print("\nMapping to BN states (minor, moderate, critical):")
    mapping = {
        "Slight Injury": "minor",
        "Serious Injury": "moderate",
        "Fatal injury": "critical",
    }

    mapped = df["Accident_severity"].map(mapping)
    mapped_counts = mapped.value_counts(normalize=True)
    print(mapped_counts)


if __name__ == "__main__":
    basic_severity_stats()
