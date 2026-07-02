from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "nhanes.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CLEAN_OUTPUT_PATH = OUTPUT_DIR / "nhanes_clean.csv"
DROPPED_OUTPUT_PATH = OUTPUT_DIR / "nhanes_dropped_rows.csv"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "nhanes_preprocessing_report.csv"


VARIABLES = [
    "Age",
    "Gender",
    "Income_Ratio",
    "BMI",
    "Waist_Circumference",
    "Vigorous_Activity",
    "Sedentary_Minutes",
    "Systolic_BP",
    "Diastolic_BP",
    "Total_Calories",
    "Protein_g",
    "Carbohydrates_g",
    "Total_Sugars_g",
    "Dietary_Fiber_g",
]


# Conservative plausibility rules.
# These are not meant to be "perfect clinical truth"; they are sanity filters.
VALIDITY_RULES = {
    "Age": lambda s: s.between(0, 120),
    "Gender": lambda s: s.isin([1, 2]),
    "Income_Ratio": lambda s: s.between(0, 10),
    "BMI": lambda s: s.between(10, 80),
    "Waist_Circumference": lambda s: s.between(40, 200),
    "Vigorous_Activity": lambda s: s.isin([0, 1]),
    "Sedentary_Minutes": lambda s: s.between(0, 1440),
    "Systolic_BP": lambda s: s.between(50, 300),
    "Diastolic_BP": lambda s: s.between(30, 200),
    "Total_Calories": lambda s: s.between(0, 15000),
    "Protein_g": lambda s: s.between(0, 600), # initially was 500, but one person has around 540 protein intake that was flagged, which is still reasonable
    "Carbohydrates_g": lambda s: s.between(0, 1500),
    "Total_Sugars_g": lambda s: s.between(0, 1000),
    "Dietary_Fiber_g": lambda s: s.between(0, 200),
}


def load_nhanes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    missing_cols = [c for c in VARIABLES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    return df


def preprocess_nhanes(df_raw: pd.DataFrame):
    df = df_raw[VARIABLES].copy()

    # Keep original row id so dropped rows can be traced back.
    df.insert(0, "original_row_index", df_raw.index)

    invalid_reasons = pd.Series("", index=df.index, dtype="object")

    # Convert selected columns to numeric.
    for col in VARIABLES:
        before_na = df[col].isna()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        became_na = df[col].isna() & ~before_na

        invalid_reasons.loc[became_na] += f"{col}: non_numeric; "

    # Missing values after conversion.
    missing_mask = df[VARIABLES].isna()
    for col in VARIABLES:
        rows = missing_mask[col]
        invalid_reasons.loc[rows] += f"{col}: missing; "

    # Explicit plausibility checks.
    for col, rule in VALIDITY_RULES.items():
        valid = rule(df[col])

        # NaNs are already handled as missing, so avoid duplicate reasons.
        invalid = (~valid) & df[col].notna()

        invalid_reasons.loc[invalid] += f"{col}: invalid_value; "

    drop_mask = invalid_reasons.str.len() > 0

    dropped = df.loc[drop_mask].copy()
    dropped["drop_reason"] = invalid_reasons.loc[drop_mask].str.strip()

    clean = df.loc[~drop_mask].copy()

    # Remove tracing column from clean output if you want a pure modeling CSV.
    clean = clean.drop(columns=["original_row_index"])

    report_rows = []

    report_rows.append({
        "check": "rows_original",
        "count": len(df_raw),
    })

    report_rows.append({
        "check": "rows_clean",
        "count": len(clean),
    })

    report_rows.append({
        "check": "rows_dropped",
        "count": len(dropped),
    })

    for col in VARIABLES:
        missing_count = df[col].isna().sum()
        invalid_count = 0

        if col in VALIDITY_RULES:
            valid = VALIDITY_RULES[col](df[col])
            invalid_count = ((~valid) & df[col].notna()).sum()

        report_rows.append({
            "check": f"{col}: missing_after_numeric_conversion",
            "count": int(missing_count),
        })

        report_rows.append({
            "check": f"{col}: invalid_value",
            "count": int(invalid_count),
        })

    report = pd.DataFrame(report_rows)

    return clean, dropped, report


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_nhanes(INPUT_PATH)

    print(f"Loaded: {INPUT_PATH}")
    print(f"Original shape: {df_raw.shape}")

    clean, dropped, report = preprocess_nhanes(df_raw)

    clean.to_csv(CLEAN_OUTPUT_PATH, index=False)
    dropped.to_csv(DROPPED_OUTPUT_PATH, index=False)
    report.to_csv(REPORT_OUTPUT_PATH, index=False)

    print(f"Clean shape: {clean.shape}")
    print(f"Dropped rows: {len(dropped)}")

    print(f"Wrote clean data to: {CLEAN_OUTPUT_PATH}")
    print(f"Wrote dropped rows to: {DROPPED_OUTPUT_PATH}")
    print(f"Wrote report to: {REPORT_OUTPUT_PATH}")

    print("\nPreprocessing report:")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()