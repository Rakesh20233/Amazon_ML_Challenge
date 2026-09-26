from pathlib import Path
import numpy as np
import pandas as pd


def calculate_entity_f05(true_matches_str: str, pred_matches_str: str) -> float:
  """Calculates the precision-weighted F_0.5 score for a single Source 1 entity."""
  true_set = (
      set(str(true_matches_str).split(","))
      if pd.notna(true_matches_str) and str(true_matches_str).strip()
      else set()
  )
  pred_set = (
      set(str(pred_matches_str).split(","))
      if pd.notna(pred_matches_str) and str(pred_matches_str).strip()
      else set()
  )

  # Singleton cases: true entity has no cross-source matches
  if not true_set:
    return 1.0 if not pred_set else 0.0

  # Non-singleton case where no matches were retrieved
  if not pred_set:
    return 0.0

  tp = len(true_set.intersection(pred_set))
  if tp == 0:
    return 0.0

  precision = tp / len(pred_set)
  recall = tp / len(true_set)

  # Official formula for beta=0.5
  return (1.25 * precision * recall) / ((0.25 * precision) + recall)


def evaluate_macro_f05(
    ground_truth_path: str, predictions_path: str
) -> float:
  """Computes macro-average F_0.5 across all Source 1 ground truth records."""
  gt_path = Path(ground_truth_path)
  pred_path = Path(predictions_path)

  if not gt_path.exists():
    raise FileNotFoundError(f"Ground truth file missing: {gt_path}")
  if not pred_path.exists():
    raise FileNotFoundError(f"Predictions file missing: {pred_path}")

  gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
  pred_df = pd.read_csv(pred_path, sep="\t", dtype=str)

  # Merge left to ensure all Source 1 entities are evaluated
  merged = pd.merge(
      gt_df,
      pred_df,
      on="source1_entity_id",
      how="left",
      suffixes=("_true", "_pred"),
  )

  # Unmatched or missing prediction rows receive empty set
  merged["matched_entity_ids_pred"] = (
      merged["matched_entity_ids_pred"].fillna("").astype(str)
  )
  merged["matched_entity_ids_true"] = (
      merged["matched_entity_ids_true"].fillna("").astype(str)
  )

  # Calculate score per entity
  scores = [
      calculate_entity_f05(t, p)
      for t, p in zip(
          merged["matched_entity_ids_true"], merged["matched_entity_ids_pred"]
      )
  ]

  macro_f05 = float(np.mean(scores))
  return macro_f05


if __name__ == "__main__":
  root_dir = Path(__file__).resolve().parents[3]
  gt = (
      root_dir
      / "dataset"
      / "processed"
      / "val_split_ground_truth.tsv"
  )
  pred = root_dir / "output" / "matching_results.tsv"

  if gt.exists() and pred.exists():
    score = evaluate_macro_f05(str(gt), str(pred))
    print(f"Validation Macro F_0.5: {score:.5f}")
  else:
    print("Metrics check: Run validation evaluation after outputs are created.")