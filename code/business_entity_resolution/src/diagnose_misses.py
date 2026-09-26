from pathlib import Path
import pandas as pd

root_dir = Path(__file__).resolve().parents[3]
train_dir = root_dir / "dataset" / "train"
artifacts_dir = root_dir / "artifacts"

print("Loading Ground Truth and Candidates sample...")
gt_df = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t")
cand_df = pd.read_parquet(artifacts_dir / "train_candidates.parquet")

# Filter to non-empty GT matches
gt_df = gt_df[
    gt_df["matched_entity_ids"].notna()
    & (gt_df["matched_entity_ids"].str.strip() != "")
]

# Explode ground truth so each (s1, target) is a row
gt_pairs = (
    gt_df.assign(
        candidate_entity_id=gt_df["matched_entity_ids"].str.split(",")
    )
    .explode("candidate_entity_id")
    .rename(columns={"source1_entity_id": "s1_id", "candidate_entity_id": "target_id"})
)

cand_pairs = set(
    zip(cand_df["source1_entity_id"], cand_df["candidate_entity_id"])
)

# Identify missed pairs
gt_pairs["is_captured"] = [
    (s, t) in cand_pairs for s, t in zip(gt_pairs["s1_id"], gt_pairs["target_id"])
]

missed = gt_pairs[~gt_pairs["is_captured"]].head(15)

# Load entity tables to inspect text
s1 = pd.read_csv(
    train_dir / "train_source1.tsv", sep="\t", index_col="entity_id"
)
s2 = pd.read_csv(
    train_dir / "train_source2.tsv", sep="\t", index_col="entity_id"
)
s3 = pd.read_csv(
    train_dir / "train_source3.tsv", sep="\t", index_col="entity_id"
)
targets = pd.concat([s2, s3])

print("\n--- SAMPLE MISSED TRUE MATCHES ---")
for idx, row in missed.iterrows():
  s1_id = row["s1_id"]
  t_id = row["target_id"]
  s1_name = s1.loc[s1_id, "business_name"] if s1_id in s1.index else "N/A"
  s1_addr = s1.loc[s1_id, "business_address"] if s1_id in s1.index else "N/A"
  t_name = targets.loc[t_id, "business_name"] if t_id in targets.index else "N/A"
  t_addr = (
      targets.loc[t_id, "business_address"] if t_id in targets.index else "N/A"
  )

  print(f"S1 [{s1_id}]: {s1_name} | {s1_addr}")
  print(f"Target [{t_id}]: {t_name} | {t_addr}")
  print("-" * 60)