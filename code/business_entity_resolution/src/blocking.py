import gc
import os
from pathlib import Path
import sys
import time

# Configure thread limits for OpenMP and native math runtimes
num_cores = os.cpu_count() or 8
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
  sys.path.append(str(SRC_DIR))

from preprocess import preprocess_dataframe


class BlockingEngine:

  def __init__(
      self,
      top_k: int = 6,
      min_sim: float = 0.35,
      batch_size: int = 50000,
      n_jobs: int = None,
  ):
    self.top_k = top_k
    self.min_sim = min_sim
    self.batch_size = batch_size
    self.n_jobs = n_jobs if n_jobs is not None else num_cores

  def _block_country_partition(
      self,
      df_s1: pd.DataFrame,
      df_targets: pd.DataFrame,
      country_name: str,
  ) -> pd.DataFrame:
    n_s1 = len(df_s1)
    n_targets = len(df_targets)

    if n_s1 == 0 or n_targets == 0:
      return pd.DataFrame(
          columns=["source1_entity_id", "candidate_entity_id", "blocking_score"]
      )

    print(f"\n=======================================================")
    print(
        f"[{country_name}] Building vocabulary on {n_targets:,} target names..."
    )
    t0_fit = time.perf_counter()

    # Pruned vocabulary: drops ultra-frequent words (max_df=0.05) and noise (min_df=4)
    # This prevents sparse columns from containing millions of non-zero entries
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=4,
        max_df=0.05,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    target_mat = vectorizer.fit_transform(df_targets["business_name_clean"])
    target_mat_T = target_mat.T.tocsr()
    print(
        f"[{country_name}] Target index built in"
        f" {time.perf_counter() - t0_fit:.1f}s | Vocab size:"
        f" {len(vectorizer.vocabulary_):,}"
    )

    s1_ids = df_s1["entity_id"].values
    target_ids = df_targets["entity_id"].values

    candidate_s1 = []
    candidate_target = []
    candidate_score = []

    num_batches = (n_s1 + self.batch_size - 1) // self.batch_size
    print(
        f"[{country_name}] Querying {n_s1:,} S1 entities across {self.n_jobs}"
        f" threads ({num_batches} batches of {self.batch_size:,})..."
    )

    t0_search = time.perf_counter()

    with tqdm(
        total=n_s1,
        desc=f"[{country_name}] Blocking",
        unit="queries",
        ncols=100,
    ) as pbar:
      for b_idx in range(num_batches):
        t_b_start = time.perf_counter()
        start_idx = b_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, n_s1)
        chunk_len = end_idx - start_idx

        # Direct in-process vectorization
        chunk_texts = df_s1["business_name_clean"].iloc[start_idx:end_idx]
        s1_vecs = vectorizer.transform(chunk_texts)

        # Threshold=0.35 cuts out shallow token collisions that flood the OpenMP priority queue
        topn_csr = sp_matmul_topn(
            s1_vecs,
            target_mat_T,
            top_n=self.top_k,
            threshold=self.min_sim,
            sort=True,
            n_threads=self.n_jobs,
        )

        topn_coo = topn_csr.tocoo()
        for r, c, s in zip(topn_coo.row, topn_coo.col, topn_coo.data):
          candidate_s1.append(s1_ids[start_idx + r])
          candidate_target.append(target_ids[c])
          candidate_score.append(float(s))

        batch_duration = time.perf_counter() - t_b_start
        qps = chunk_len / max(batch_duration, 1e-4)

        pbar.update(chunk_len)
        pbar.set_postfix({
            "batch": f"{b_idx + 1}/{num_batches}",
            "cand": f"{len(candidate_s1):,}",
            "speed": f"{int(qps):,} q/s",
        })

    total_time = time.perf_counter() - t0_search
    print(
        f"[{country_name}] Complete in {total_time:.1f}s | Average speed:"
        f" {int(n_s1 / max(total_time, 1e-4)):,} queries/sec | Candidates:"
        f" {len(candidate_s1):,}"
    )

    del target_mat, target_mat_T
    gc.collect()

    return pd.DataFrame({
        "source1_entity_id": candidate_s1,
        "candidate_entity_id": candidate_target,
        "blocking_score": candidate_score,
    })

  def generate_candidate_pairs(
      self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame
  ) -> pd.DataFrame:
    print("[Preprocessing] Normalizing names across datasets...")
    t0_prep = time.perf_counter()

    df_s1 = preprocess_dataframe(df_s1)
    df_s2 = preprocess_dataframe(df_s2)
    df_s3 = preprocess_dataframe(df_s3)

    print(
        f"[Preprocessing] Normalization completed in"
        f" {time.perf_counter() - t0_prep:.1f}s"
    )

    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

    all_candidates = []
    unique_countries = df_s1["country"].dropna().unique()

    for country in unique_countries:
      sub_s1 = df_s1[df_s1["country"] == country]
      sub_targets = df_targets[df_targets["country"] == country]

      part_cand = self._block_country_partition(
          sub_s1, sub_targets, str(country)
      )
      all_candidates.append(part_cand)

    if all_candidates:
      df_result = pd.concat(all_candidates, ignore_index=True)
    else:
      df_result = pd.DataFrame(
          columns=["source1_entity_id", "candidate_entity_id", "blocking_score"]
      )

    return df_result.drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"]
    )


if __name__ == "__main__":
  root_dir = Path(__file__).resolve().parents[3]
  train_dir = root_dir / "dataset" / "train"

  if (train_dir / "train_source1.tsv").exists():
    print("Testing BlockingEngine on sample batch...")
    sample_s1 = pd.read_csv(
        train_dir / "train_source1.tsv", sep="\t", nrows=10000
    )
    sample_s2 = pd.read_csv(
        train_dir / "train_source2.tsv", sep="\t", nrows=20000
    )
    sample_s3 = pd.read_csv(
        train_dir / "train_source3.tsv", sep="\t", nrows=20000
    )

    engine = BlockingEngine(top_k=6, min_sim=0.35, batch_size=5000)
    res = engine.generate_candidate_pairs(sample_s1, sample_s2, sample_s3)
    print(f"Generated {len(res):,} sample candidates.")