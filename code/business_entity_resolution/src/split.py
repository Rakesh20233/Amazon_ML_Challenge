import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_validation_split(
    val_size: float = 0.20,
    random_seed: int = 42,
):
  """Creates a stratified holdout split, saves artifacts, and dumps val_split_ids.json."""
  root_dir = Path(__file__).resolve().parents[3]
  train_dir = root_dir / "dataset" / "train"
  processed_dir = root_dir / "dataset" / "processed"
  artifacts_dir = root_dir / "artifacts"

  processed_dir.mkdir(parents=True, exist_ok=True)
  artifacts_dir.mkdir(parents=True, exist_ok=True)

  s1_file = train_dir / "train_source1.tsv"
  gt_file = train_dir / "train_ground_truth.tsv"

  if not s1_file.exists() or not gt_file.exists():
    raise FileNotFoundError(f"Missing train files under {train_dir}")

  print("Loading Source 1 and Ground Truth data...")
  s1_df = pd.read_csv(s1_file, sep="\t")
  gt_df = pd.read_csv(gt_file, sep="\t")

  # Stratify by country to preserve regional distributions
  stratify_col = s1_df["country"].fillna("UNKNOWN")

  s1_train, s1_val = train_test_split(
      s1_df,
      test_size=val_size,
      random_state=random_seed,
      stratify=stratify_col,
  )

  val_entity_ids = sorted(list(s1_val["entity_id"].astype(str).unique()))
  train_entity_ids = set(s1_train["entity_id"].astype(str))

  # Export the validation entity IDs to JSON
  val_ids_path = artifacts_dir / "val_split_ids.json"
  with open(val_ids_path, "w", encoding="utf-8") as f:
    json.dump(val_entity_ids, f, indent=2)
  print(
      f"Exported {len(val_entity_ids):,} validation Source 1 IDs to"
      f" {val_ids_path}"
  )

  # Filter Ground Truth matches
  gt_train = gt_df[gt_df["source1_entity_id"].astype(str).isin(train_entity_ids)]
  gt_val = gt_df[gt_df["source1_entity_id"].astype(str).isin(set(val_entity_ids))]

  print(
      f"Training S1: {len(s1_train):,} | Validation S1: {len(s1_val):,} (20%)"
  )

  # Save table splits
  s1_train.to_csv(
      processed_dir / "train_split_source1.tsv", sep="\t", index=False
  )
  s1_val.to_csv(processed_dir / "val_split_source1.tsv", sep="\t", index=False)
  gt_train.to_csv(
      processed_dir / "train_split_ground_truth.tsv", sep="\t", index=False
  )
  gt_val.to_csv(
      processed_dir / "val_split_ground_truth.tsv", sep="\t", index=False
  )
  print(f"Dataframe splits saved under: {processed_dir}")


if __name__ == "__main__":
  create_stratified_validation_split()