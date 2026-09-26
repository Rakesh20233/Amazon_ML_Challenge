import os
import pandas as pd
from blocking import BlockingEngine

def export_candidate_pairs_tsv(df_all_s1: pd.DataFrame, df_candidates: pd.DataFrame, output_path: str):
    """
    Exports candidates to candidate_pairs.tsv matching all competition requirements:
    - Tab separated
    - Exactly one row per Source 1 entity
    - Singletons have empty string
    - Comma-separated list with no duplicate IDs
    """
    # Group candidate IDs by source1_entity_id
    cand_map = (
        df_candidates.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(list(dict.fromkeys(ids)))) # preserves order & dedups
        .to_dict()
    )

    records = []
    for s1_id in df_all_s1["entity_id"]:
        matched_str = cand_map.get(s1_id, "")
        records.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": matched_str
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"Successfully wrote {len(out_df)} rows to {output_path}")

def run_test_generation():
    print("Loading test datasets...")
    df_s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")
    df_s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t")
    df_s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t")

    print(f"Test records: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}")

    # Initialize blocking engine (Top-K=6, min_similarity=0.22)
    engine = BlockingEngine(top_k=6, min_sim=0.22)
    
    print("Running blocking across countries (US, India, France)...")
    cand_df = engine.generate_candidate_pairs(df_s1, df_s2, df_s3)
    
    avg_cand = len(cand_df) / len(df_s1) if len(df_s1) > 0 else 0
    print(f"Total candidate pairs: {len(cand_df)}")
    print(f"Average candidates per S1 entity: {avg_cand:.2f}")

    # 1. Save artifact for Person 3
    os.makedirs("artifacts", exist_ok=True)
    cand_df.to_parquet("artifacts/test_candidates.parquet", index=False)
    print("Saved candidates to artifacts/test_candidates.parquet")

    # 2. Export competition TSV
    os.makedirs("output", exist_ok=True)
    export_candidate_pairs_tsv(df_s1, cand_df, "output/candidate_pairs.tsv")

if __name__ == "__main__":
    run_test_generation()