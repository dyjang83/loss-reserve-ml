import pandas as pd
from pathlib import Path

# Path to your raw data
RAW_DIR = Path(__file__).parents[2] / "data" / "raw"

# Matches your renamed files
LINE_FILES = {
    "ppauto":   "ppauto_pos.csv",
    "comauto":  "comauto_pos.csv",
    "wkcomp":   "wkcomp_pos.csv",
    "medmal":   "medmal_pos.csv",
    "othliab":  "othliab_pos.csv",
    "prodliab": "prodliab_pos.csv",
}

def load_line(line: str) -> pd.DataFrame:
    path = RAW_DIR / LINE_FILES[line]
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    df = pd.read_csv(path)
    df["line"] = line
    return df

def load_all() -> pd.DataFrame:
    return pd.concat([load_line(l) for l in LINE_FILES], ignore_index=True)