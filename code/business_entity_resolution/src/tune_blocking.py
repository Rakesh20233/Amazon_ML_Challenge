import sys
from pathlib import Path
import pandas as pd

# Automatically anchors to the root directory 'amazon_challenge'
# File is at: amazon_challenge/code/business_entity_resolution/src/tune_blocking.py
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(Path(__file__).resolve().parent))

from blocking import BlockingEngine

def evaluate_blocking(top_k=6, min_sim=0.15):
    print(f"--- Running Blocking Evaluation (top_k={top_k}, min_sim={min_sim}) ---")
    print(f"Project root identified as: {ROOT_DIR}")

    train_dir = ROOT_DIR / "dataset" / "train"
    print(f"Loading data from: {train_dir}")
    
    # Read files with explicit tab separator
    s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t")
    s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t")
    s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t")
    gt = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t")

    print(f"Loaded: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}")

    # 1. Run blocking
    engine = BlockingEngine(top_k=top_k, min_sim=min_sim)
    candidates = engine.generate_candidate_pairs(s1, s2, s3)

    # 2. Extract true pairs from ground truth
    gt_pairs = set()
    for _, r in gt.iterrows():
        val = r["matched_entity_ids"]
        if pd.notna(val) and str(val).strip():
            for m in str(val).split(","):
                gt_pairs.add((r["source1_entity_id"], m.strip()))

    # 3. Calculate candidate recall & average candidates per S1
    cand_pairs = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    captured = len(gt_pairs.intersection(cand_pairs))
    recall = (captured / len(gt_pairs)) * 100 if len(gt_pairs) > 0 else 0
    avg_cand = len(candidates) / len(s1) if len(s1) > 0 else 0

    print("\n================ RESULTS ================")
    print(f"Total True Positive Pairs: {len(gt_pairs)}")
    print(f"Captured by Candidates:    {captured}")
    print(f"Candidate Recall Ceiling:  {recall:.2f}%")
    print(f"Avg Candidates / S1:       {avg_cand:.2f}")
    print("=========================================\n")

    # 4. Save training candidates for Person 3
    artifacts_dir = ROOT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifacts_dir / "train_candidates.parquet"
    candidates.to_parquet(out_file, index=False)
    print(f"Saved candidate pool to {out_file} for Person 3.")

if __name__ == "__main__":
  evaluate_blocking(top_k=6, min_sim=0.20)