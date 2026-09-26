# Amazon_ML_Challenge

## File Structure to follow
 
```txt
amazon_ml_challenge_2026/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── utils/
│   └── validate_submission.py       # Validation script provided in student_resource/
├── artifacts/                       # Git-ignored cache folder for handoffs
│   ├── val_split_ids.json           # Person 1: Validation S1 entity IDs
│   ├── train_candidates.parquet     # Person 2: Training candidate pairs for ML
│   ├── test_candidates.parquet      # Person 2: Test candidate pairs for ML
│   ├── lgb_model.txt                # Person 3: Trained LightGBM model weights
│   └── test_scored_pairs.parquet    # Person 3: Probabilities for test candidate pairs
├── output/                          # Target directory for submission outputs
│   ├── matching_results.tsv         # Leaderboard predictions
│   └── candidate_pairs.tsv          # Blocking candidate outputs
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py            # Global paths, random seeds, top-k constants
│       │   ├── preprocess.py        # Token cleaners & abbreviation normalizers
│       │   ├── evaluate.py          # Strict Macro F_0.5 evaluator including singletons
│       │   ├── blocking.py          # Country partition + TF-IDF character n-gram index
│       │   ├── generate_candidates.py # Runner to build candidate_pairs.tsv
│       │   ├── features.py          # RapidFuzz similarities, numeric address match
│       │   ├── train.py             # LightGBM pairwise binary classification training
│       │   ├── infer.py             # Batch candidate pair feature scoring
│       │   └── postprocess.py       # F_0.5 thresholding & matches ⊆ candidates check
│       ├── main.py                  # End-to-end execution entry point
│       ├── README.md                # Setup & reproduction guide
│       └── requirements.txt         # Pinned pip dependencies
└── Documentation_template.md        # Filled methodology write-up

```