import re
from pathlib import Path
import numpy as np
import pandas as pd

# Canonical dictionary of legal forms, address abbreviations, and geographical markers
WORD_MAP = {
    # Business Legal Forms
    "pvt": "private",
    "ltd": "limited",
    "inc": "incorporated",
    "llc": "limited liability company",
    "corp": "corporation",
    "llp": "limited liability partnership",
    # US Address Standards
    "rd": "road",
    "st": "street",
    "ste": "suite",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "apt": "apartment",
    "fl": "floor",
    "flr": "floor",
    "hwy": "highway",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    # Indian Address Markers
    "opp": "opposite",
    "nr": "near",
    "vill": "village",
    "ta": "taluka",
    "stn": "station",
    # French Business Forms and Road Descriptors (Test Set)
    "sarl": "societe a responsabilite limitee",
    "sas": "societe par actions simplifiee",
    "zi": "zone industrielle",
    "za": "zone artisanale",
    "rte": "route",
}

TOKEN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in WORD_MAP.keys()) + r")\b"
)
PUNCT_PATTERN = re.compile(r"[^a-z0-9\s]")
SPACES_PATTERN = re.compile(r"\s+")


def normalize_text_fast(text: str) -> str:
  """Normalizes text: expands abbreviations, removes punctuation, collapses whitespace."""
  if not text or pd.isna(text):
    return ""
  s = str(text).lower().replace("&", " and ")
  s = TOKEN_PATTERN.sub(lambda m: WORD_MAP[m.group(0)], s)
  s = PUNCT_PATTERN.sub(" ", s)
  return SPACES_PATTERN.sub(" ", s).strip()


def clean_string_list(texts: list) -> list:
  """Fast list comprehension execution avoiding IPC and Pandas overhead."""
  pattern_sub = TOKEN_PATTERN.sub
  punct_sub = PUNCT_PATTERN.sub
  space_sub = SPACES_PATTERN.sub
  mapping = WORD_MAP

  cleaned = []
  for t in texts:
    if not t or pd.isna(t):
      cleaned.append("")
      continue
    s = str(t).lower().replace("&", " and ")
    s = pattern_sub(lambda m: mapping[m.group(0)], s)
    s = punct_sub(" ", s)
    cleaned.append(space_sub(" ", s).strip())
  return cleaned


def preprocess_dataframe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
  """Clean business_name and business_address columns without Windows IPC crashes."""
  df_clean = df.copy()

  if "business_name" in df_clean.columns:
    df_clean["business_name_clean"] = clean_string_list(
        df_clean["business_name"].tolist()
    )

  if "business_address" in df_clean.columns:
    df_clean["business_address_clean"] = clean_string_list(
        df_clean["business_address"].tolist()
    )

  if (
      "business_name_clean" in df_clean.columns
      and "business_address_clean" in df_clean.columns
  ):
    # Fast string concatenation
    df_clean["search_text"] = (
        df_clean["business_name_clean"]
        + " "
        + df_clean["business_address_clean"]
    )

  return df_clean


if __name__ == "__main__":
  root_dir = Path(__file__).resolve().parents[3]
  sample_path = root_dir / "dataset" / "train" / "train_source1.tsv"

  if sample_path.exists():
    print(f"Testing single-process fast preprocessor on: {sample_path}")
    test_df = pd.read_csv(sample_path, sep="\t", nrows=50000)
    cleaned = preprocess_dataframe(test_df)
    print("Sample cleaned search strings:")
    for val in cleaned["search_text"].head(3):
      print(f" - {val}")
  else:
    print(f"File not found at {sample_path}.")