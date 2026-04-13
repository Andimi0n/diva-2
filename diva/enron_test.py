import os
import logging
import argparse
from enum import Enum
import numpy as np
import pandas as pd
import joblib
from pymfe.mfe import MFE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from aim import Run, Image
from enron_benchmark.alfa import alfa_poison

# --- Enumerations for Strict Typing ---
class DatasetOptions(Enum):
    ENRON = "enron"
    IMDB = "imdb"

class MethodOptions(Enum):
    ALFA = "alfa"
    ALFA_POISON = "alfa_poison"
    RANDOM = "random"
    FEATURE_NOISE = "feature_noise"
    POISSVM = "poissvm_svm"

def setup_logger(dataset_name, method_name):
    """Dynamically creates a logger based on parsed arguments."""
    os.makedirs("./logs", exist_ok=True)
    logger = logging.getLogger("DIVA_Pipeline")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs if running in interactive environments
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    fh = logging.FileHandler(f"./logs/pipeline_{dataset_name}_{method_name}.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def poison_data(X, y, rate, method, logger, clf=None):
    """ Poisons a dataset directly in RAM. """
    if rate == 0.0:
        return X.copy(), y.copy()

    if method == MethodOptions.RANDOM:
        n_flip = int(len(y) * rate)
        flip_indices = np.random.choice(len(y), n_flip, replace=False)
        y_poisoned = y.copy()
        y_poisoned[flip_indices] = -y_poisoned[flip_indices]
        
    elif method == MethodOptions.ALFA:
        from enron_benchmark.alfa import alfa_pytorch
        logger.info(f"Executing PyTorch ALFA optimization at rate {rate:.2f}")
        y_poisoned = alfa_pytorch(X, y, rate, max_iter=5)
    
    elif method == MethodOptions.ALFA_POISON:
        logger.info(f"Executing Heuristic ALFA poison at rate {rate:.2f}")
        _, y_poisoned = alfa_poison(X, y, epsilon=rate, logger=logger)
        
    else:
        logger.error(f"Poisoning method '{method.value}' is not implemented for RAM generation.")
        raise NotImplementedError(f"Poisoning method '{method.value}' is not implemented.")

    return X.copy(), y_poisoned


def step1_generate_poisoned_csvs(args, paths, logger):
    logger.info("--- STEP 1: Generating Poisoned CSVs ---")
    os.makedirs(paths['csv_output_dir'], exist_ok=True)
    
    logger.info(f"Loading raw data from {paths['npz_path']}")
    f = np.load(paths['npz_path'], allow_pickle=True, encoding='latin1')
    X_train_sparse = f['X_train'].reshape(1)[0]
    y_train = np.where(f['Y_train'] > 0, 1, -1)

    # 1. Handle Dimensionality Reduction & Sparsity
    if args.method in [MethodOptions.ALFA, MethodOptions.ALFA_POISON]:
        logger.info(f"Applying TruncatedSVD (n_components={args.svd_components}) for ALFA...")
        svd = TruncatedSVD(n_components=args.svd_components, random_state=42)
        X_train_dense = svd.fit_transform(X_train_sparse) 
        
        joblib.dump(svd, paths['svd_model_path'])
        logger.info(f"Saved SVD model to {paths['svd_model_path']}")
        
        # 2. Handle Downsampling
        if X_train_dense.shape[0] > args.max_sample:
            logger.info(f"Downsampling from {X_train_dense.shape[0]} to {args.max_sample} samples...")
            X_train_dense, y_train = resample(
                X_train_dense, y_train, 
                n_samples=args.max_sample, 
                stratify=y_train, 
                random_state=42
            )
    else:
        X_train_dense = X_train_sparse.toarray() if sp.issparse(X_train_sparse) else X_train_sparse

    advx_range = np.arange(0.0, 0.41, 0.05)
    
    for rate in advx_range:
        file_path = os.path.join(paths['csv_output_dir'], f"{args.dataset.value}_{args.method.value}_noise_{rate:.2f}.csv")
        logger.info(f"Injecting poisoning at rate {rate:.2f}...")
        
        X_noisy, y_train_pois = poison_data(X_train_dense, y_train, rate, method=args.method, logger=logger)
        
        df = pd.DataFrame(X_noisy)
        df['target_label'] = y_train_pois
        df.to_csv(file_path, index=False)
        logger.info(f"Saved poisoned dataset: {file_path}")


def step2_evaluate_diva(args, paths, logger, aim_run):
    logger.info("--- STEP 2: DIVA Evaluation ---")
    f = np.load(paths['npz_path'], allow_pickle=True, encoding='latin1')
    X_test_sparse = f['X_test'].reshape(1)[0]
    y_test = np.where(f['Y_test'] > 0, 1, -1)
    
    if args.method in [MethodOptions.ALFA, MethodOptions.ALFA_POISON] and os.path.exists(paths['svd_model_path']):
        logger.info("Loading SVD model to transform test data...")
        svd = joblib.load(paths['svd_model_path'])
        X_test_dense = svd.transform(X_test_sparse)
    else:
        X_test_dense = X_test_sparse.toarray() if sp.issparse(X_test_sparse) else X_test_sparse
    
    logger.info(f"Loading Meta-Learner from {paths['model_path']}")
    meta_learner = joblib.load(paths['model_path'])
    advx_range = np.arange(0.0, 0.41, 0.05)
    
    results_empirical, results_predicted, results_flags = [], [], []
    
    for rate in advx_range:
        file_path = os.path.join(paths['csv_output_dir'], f"{args.dataset.value}_{args.method.value}_noise_{rate:.2f}.csv")
        logger.debug(f"Evaluating file: {file_path}")
        df = pd.read_csv(file_path)
        y_train_pois = df['target_label'].values
        X_train_pois = df.drop('target_label', axis=1).values
        
        # Train and Evaluate
        clf = SVC(kernel='rbf')
        clf.fit(X_train_pois, y_train_pois)
        acc_emp = accuracy_score(y_test, clf.predict(X_test_dense))
        
        # Meta-feature extraction
        X_sample, y_sample = resample(
            X_train_pois, y_train_pois, 
            n_samples=1000, 
            stratify=y_train_pois,
            random_state=42
        )
        mfe = MFE(groups=["complexity"], suppress_warnings=True)
        mfe.fit(X_sample, y_sample)
        ft, vals = mfe.extract()
        
        C_measures = pd.DataFrame([dict(zip(ft, vals))])
        
        X_meta = C_measures.fillna(0.0).values
        acc_pred = meta_learner.predict(X_meta)[0]
        
        # DIVA Flagging Logic
        is_flagged = abs(acc_emp - acc_pred) > (acc_emp * 0.05)
        
        results_empirical.append(acc_emp)
        results_predicted.append(acc_pred)
        results_flags.append(is_flagged)
        na_count = C_measures.isna().sum().sum()

        logger.info(f"Rate {rate:.2f} | Emp: {acc_emp:.4f} | Pred: {acc_pred:.4f} | Flag: {is_flagged} | Meta-NAs: {na_count}")

        # --- AIM METRIC TRACKING ---
        step_idx = int(rate * 100) 
        aim_run.track(acc_emp, name='accuracy', context={'type': 'empirical'}, step=step_idx)
        aim_run.track(acc_pred, name='accuracy', context={'type': 'predicted'}, step=step_idx)
        aim_run.track(float(is_flagged), name='diva_flag_triggered', step=step_idx)
        aim_run.track(na_count, name='meta_features_na_count', step=step_idx)
        aim_run.track(rate, name='poisoning_rate', step=step_idx)

    visualize_diva(args, advx_range, results_empirical, results_predicted, results_flags, logger, aim_run)


def visualize_diva(args, rates, empirical, predicted, flags, logger, aim_run):
    logger.info("Generating DIVA Detection Plot...")
    fig = plt.figure(figsize=(10, 6))
    plt.plot(rates, empirical, 'o-', label='Empirical Acc (Poisoned)')
    plt.plot(rates, predicted, 's--', label='Predicted Acc (Clean Expectation)')
    
    for i, flag in enumerate(flags):
        if flag:
            plt.axvspan(rates[i]-0.02, rates[i]+0.02, color='red', alpha=0.1, label='DIVA Detection' if i==0 else "")
            
    plt.title(f"DIVA Detection: Empirical vs Predicted Accuracy ({args.method.value.upper()})")
    plt.xlabel("Poisoning Rate")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0,1)

    # --- AIM IMAGE TRACKING ---
    aim_image = Image(fig)
    aim_run.track(aim_image, name='diva_detection_plot', context={'dataset': args.dataset.value})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIVA Poisoning and Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, choices=[e.value for e in DatasetOptions], default=DatasetOptions.ENRON.value, 
                        help="The dataset to process (e.g., enron, imdb)")
    parser.add_argument("--method", type=str, choices=[e.value for e in MethodOptions], default=MethodOptions.ALFA_POISON.value, 
                        help="The poisoning method to apply")
    parser.add_argument("--max_sample", type=int, default=50000, 
                        help="Maximum number of samples allowed after downsampling")
    parser.add_argument("--svd_components", type=int, default=100, 
                        help="Number of components for TruncatedSVD dimensionality reduction")
    parser.add_argument("--description", type=str, default="", 
                        help="Description of this run to be stored in Aim")
    
    args = parser.parse_args()

    args.dataset = DatasetOptions(args.dataset)
    args.method = MethodOptions(args.method)

    # --- Dynamic Paths ---
    paths = {
        'npz_path': f"./data/test/{args.dataset.value}/{args.dataset.value}_processed_sparse.npz",
        'model_path': "./data/metalearners/metalearner_feature_noise_svm+random_flip_svm+poissvm_svm+alfa_svm.pkl",
        'csv_output_dir': f"./data/test/{args.dataset.value}/{args.dataset.value}_poisoned_csvs/",
        'svd_model_path': f"./data/test/{args.dataset.value}/{args.dataset.value}_svd_model.pkl"
    }

    logger = setup_logger(args.dataset.value, args.method.value)
    logger.info("Starting Poisoning and Evaluation Pipeline")
    
    # --- Initialize Aim Run ---
    run = Run(experiment="DIVA_Evaluation_Pipeline")
    run.set("description", args.description)
    
    # Log Hyperparameters
    run["hparams"] = {
        "dataset": args.dataset.value,
        "method": args.method.value,
        "max_sample": args.max_sample,
        "svd_components": args.svd_components,
        "description": args.description
    }

    try:
        step1_generate_poisoned_csvs(args, paths, logger)
        step2_evaluate_diva(args, paths, logger, aim_run=run)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.exception("A fatal error occurred during pipeline execution:")
    finally:
        run.close()