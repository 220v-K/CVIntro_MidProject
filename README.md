# CIFAR-10 KNN Experiment Project

## Overview
This project benchmarks several evaluation strategies for the K-Nearest Neighbors (KNN) classifier on the CIFAR-10 image classification task. Three experiment modes are supported—`train/test` split, `train/valid/test` split, and `5-fold` cross-validation—while sharing the same hyperparameter search space (k ∈ {1,3,5,7,9}, distance metrics L1/L2). Each run gathers accuracy, precision, recall, and F1 score, and stores the metrics as logs and plots.

## Folder Structure
```text
CVIntro_MidProject/
├─ cifar-10-batches-py/          # CIFAR-10 original batch files (Python version)
├─ main.py                       # Experiment entry point; utilities and data loader
├─ train_test_split_only.py      # KNN experiment with a simple train/test split
├─ train_valid_test_split.py     # Train/valid/test split plus final test evaluation with best hyperparameters
├─ five_fold_cross_validation.py # 5-fold StratifiedKFold experiment
├─ plot.py                       # Metric visualization and results directory management
├─ results/                      # Result logs and figures generated after running experiments
└─ README.md
```

## Requirements
- Python 3.10 or newer is recommended.
- Required packages: `numpy`, `scikit-learn`, `pandas`, `tqdm`, `matplotlib`
  - Example (with virtual environment): `pip install numpy scikit-learn pandas tqdm matplotlib`
- The CIFAR-10 Python batch files must be under `cifar-10-batches-py/`. If the files are already included in the repository, no additional download is needed.

## How to Run
1. (Optional) Create a virtual environment and install dependencies.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy scikit-learn pandas tqdm matplotlib
   ```
2. Choose an experiment mode with the `--classifier` argument.
   ```bash
   python main.py --classifier train_test       # plain train/test split
   python main.py --classifier train_valid_test # train/valid/test split
   python main.py --classifier 5-fold           # 5-fold cross-validation
   ```
3. Each experiment iterates over every k (1,3,5,7,9) and distance metric (L1, L2), training and evaluating the KNN models. Progress is shown with a `tqdm` progress bar.

## Experiment Modes
| Argument | Description | Validation Strategy | Test Evaluation with Best Hyperparameters |
|----------|-------------|---------------------|-------------------------------------------|
| `train_test` | Train on the full training set and evaluate on the test set | Single split | Immediate test evaluation |
| `train_valid_test` | Split the training set into 90% train and 10% validation | Fixed validation split | Retrain with the best combo, then evaluate on the test set |
| `5-fold` | Perform 5-fold StratifiedKFold and average the results | Cross-validation | Retrain on the full training set with the best combo, then test |

## Key Scripts and Functions
- `main.py`
  - `load_data()` / `load_data_train_valid_test()`: Load CIFAR-10 batches and build train/validation/test sets.
  - `run_knn(case)`: Dispatch each split strategy, store results, and trigger plotting.
  - `save_results_to_txt()`: Log metrics per configuration, summarize the best scores, and record test-set results.
- `train_test_split_only.py`
  - `knn_classifier_train_test_split_only()`: Train and evaluate KNN for a given k and distance metric.
- `train_valid_test_split.py`
  - `knn_classifier_train_valid_test_split()`: Search hyperparameters using the validation set.
  - `evaluate_with_best_k_and_distance_metric()`: Retrain with the best configuration and evaluate on the test set.
- `five_fold_cross_validation.py`
  - `knn_classifier_5_fold_cross_validation()`: Average metrics across five StratifiedKFold splits.
- `plot.py`
  - `plot_results()`: Save accuracy, precision, recall, and F1 curves per distance metric.
  - `ensure_results_dir()`: Create the results directory when needed.

## Viewing Results
- Each mode writes outputs under `results/<experiment_name>/`.
  - `knn_results.txt`: Metrics per k and distance metric, the best configuration summary, and test-set evaluation when applicable.
  - `knn_l1_distance.png`, `knn_l2_distance.png`: Performance curves by distance metric.
- Running the same experiment name multiple times overwrites existing files; back up the `results` directory if you need to keep earlier runs.

This README was written with GPT-5-Codex. Code is hand-written.
