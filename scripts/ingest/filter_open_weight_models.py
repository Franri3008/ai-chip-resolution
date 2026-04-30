"""Filter Epoch AI all_ai_models.csv to open-weight models with known training hardware.

Reads database/all_ai_models.csv, keeps rows where Model accessibility starts with
"Open weights" and Training hardware is non-empty, projects to a hardware-relevant
column subset (including Confidence), and writes database/open_weight_models_with_hardware.csv.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "database" / "all_ai_models.csv"
DST = ROOT / "database" / "open_weight_models_with_hardware.csv"

RELEVANT_COLUMNS = [
    "Model",
    "Organization",
    "Organization categorization",
    "Country (of organization)",
    "Authors",
    "Publication date",
    "Domain",
    "Task",
    "Reference",
    "Link",
    "Hugging Face developer id",
    "Model accessibility",
    "Open model weights?",
    "Training code accessibility",
    "Accessibility notes",
    "Confidence",
    "Training hardware",
    "Hardware quantity",
    "Training chip-hours",
    "Training cloud compute vendor",
    "Training data center",
    "Hardware utilization (MFU)",
    "Hardware utilization (HFU)",
    "Utilization notes",
    "Training power draw (W)",
    "Numerical format",
    "Parameters",
    "Training compute (FLOP)",
    "Training compute lower bound",
    "Training compute upper bound",
    "Training compute notes",
    "Training compute estimation method",
    "Training compute cost (2023 USD)",
    "Training time (hours)",
    "Training dataset size (total)",
    "Epochs",
    "Batch size",
    "Frontier model",
    "Foundation model",
    "Notability criteria",
    "Last modified",
]


def main() -> None:
    df = pd.read_csv(SRC)

    accessibility = df["Model accessibility"].fillna("")
    hardware = df["Training hardware"].fillna("").astype(str).str.strip()

    is_open_weight = accessibility.str.startswith("Open weights")
    has_hardware = hardware != ""
    filtered = df.loc[is_open_weight & has_hardware, RELEVANT_COLUMNS].copy()

    filtered = filtered.sort_values(
        by=["Publication date", "Model"], ascending=[False, True], na_position="last"
    )

    DST.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(DST, index=False)

    print(f"Source rows:   {len(df):>5}")
    print(f"Filtered rows: {len(filtered):>5}")
    print(f"Wrote: {DST.relative_to(ROOT)}")
    print()
    print("Confidence breakdown:")
    print(filtered["Confidence"].value_counts(dropna=False).to_string())
    print()
    print("Accessibility breakdown:")
    print(filtered["Model accessibility"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
