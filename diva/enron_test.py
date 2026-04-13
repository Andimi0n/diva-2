import os
import logging
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

# Assuming alfa_poison is in your local modules
from enron_benchmark.alfa import alfa_poison

# --- Configuration ---
DATASET = 'enron'
METHOD = 'alfa_poison'
MAX_SAMPLE = 50000

ENRON_NPZ_PATH = f"./data/{DATASET}_processed_sparse.npz"
MODEL_PATH = "./results/metalearners/metalearner_feature_noise_svm+random_flip_svm+poissvm_svm+alfa_svm.pkl" 
CSV_OUTPUT_DIR = f"./data/{DATASET}_poisoned_csvs/"
SVD_MODEL_PATH = f"./data/{DATASET}_svd_model.pkl"

# --- Logging Setup ---
# Creates a log file specifically for the chosen dataset and outputs to the console
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"./logs/pipeline_{DATASET}_{METHOD}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def poison_data(X, y, rate, method='random', clf=None):
    """
    Poisons a dataset directly in RAM.
    """
    if rate == 0.0:
        return X.copy(), y.copy()

    if method == 'random':
        n_flip = int(len(y) * rate)
        flip_indices = np.random.choice(len(y), n_flip, replace=False)
        y_poisoned = y.copy()
        y_poisoned[flip_indices] = -y_poisoned[flip_indices]
        
    elif method == 'alfa':
        from enron_benchmark.alfa import alfa_pytorch
        logger.info(f"Executing PyTorch ALFA optimization at rate {rate:.2f}")
        y_poisoned = alfa_pytorch(X, y, rate, max_iter=5)
    
    elif method == 'alfa_poison':
        logger.info(f"Executing Heuristic ALFA poison at rate {rate:.2f}")
        _, y_poisoned = alfa_poison(X, y, epsilon=rate, logger = logger)
        
    else:
        logger.error(f"Poisoning method '{method}' is not implemented.")
        raise NotImplementedError(f"Poisoning method '{method}' is not implemented.")

    return X.copy(), y_poisoned


def step1_generate_poisoned_csvs():
    logger.info("--- STEP 1: Generating Poisoned CSVs ---")
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Loading raw data from {ENRON_NPZ_PATH}")
    f = np.load(ENRON_NPZ_PATH, allow_pickle=True, encoding='latin1')
    X_train_sparse = f['X_train'].reshape(1)[0]
    y_train = np.where(f['Y_train'] > 0, 1, -1)

    # 1. Handle Dimensionality Reduction & Sparsity
    if METHOD in ['alfa', 'alfa_poison']:
        logger.info("Applying TruncatedSVD (n_components=100) for ALFA...")
        svd = TruncatedSVD(n_components=100, random_state=42)
        X_train_dense = svd.fit_transform(X_train_sparse) 
        
        # Save SVD so the test set can be transformed in Step 2
        joblib.dump(svd, SVD_MODEL_PATH)
        logger.info(f"Saved SVD model to {SVD_MODEL_PATH}")
        
        # 2. Handle Downsampling
        if X_train_dense.shape[0] > MAX_SAMPLE:
            logger.info(f"Downsampling from {X_train_dense.shape[0]} to {MAX_SAMPLE} samples...")
            X_train_dense, y_train = resample(
                X_train_dense, y_train, 
                n_samples=MAX_SAMPLE, 
                stratify=y_train, 
                random_state=42
            )
    else:
        # If not ALFA, just convert to dense
        X_train_dense = X_train_sparse.toarray() if sp.issparse(X_train_sparse) else X_train_sparse

    advx_range = np.arange(0.0, 0.41, 0.05)
    
    for rate in advx_range:
        file_path = os.path.join(CSV_OUTPUT_DIR, f"{DATASET}_{METHOD}_noise_{rate:.2f}.csv")
        logger.info(f"Injecting poisoning at rate {rate:.2f}...")
        
        X_noisy, y_train_pois = poison_data(X_train_dense, y_train, rate, method=METHOD)
        
        df = pd.DataFrame(X_noisy)
        df['target_label'] = y_train_pois
        df.to_csv(file_path, index=False)
        logger.info(f"Saved poisoned dataset: {file_path}")


def step2_evaluate_diva():
    logger.info("--- STEP 2: DIVA Evaluation ---")
    f = np.load(ENRON_NPZ_PATH, allow_pickle=True, encoding='latin1')
    X_test_sparse = f['X_test'].reshape(1)[0]
    y_test = np.where(f['Y_test'] > 0, 1, -1)
    
    # Match Test Set dimensions to Train Set dimensions
    if METHOD in ['alfa', 'alfa_poiso'] and os.path.exists(SVD_MODEL_PATH):
        logger.info("Loading SVD model to transform test data...")
        svd = joblib.load(SVD_MODEL_PATH)
        X_test_dense = svd.transform(X_test_sparse)
    else:
        X_test_dense = X_test_sparse.toarray() if sp.issparse(X_test_sparse) else X_test_sparse
    
    logger.info(f"Loading Meta-Learner from {MODEL_PATH}")
    meta_learner = joblib.load(MODEL_PATH)
    advx_range = np.arange(0.0, 0.41, 0.05)
    
    results_empirical, results_predicted, results_flags = [], [], []
    
    for rate in advx_range:
        file_path = os.path.join(CSV_OUTPUT_DIR, f"{DATASET}_{METHOD}_noise_{rate:.2f}.csv")
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

    visualize_diva(advx_range, results_empirical, results_predicted, results_flags)


def visualize_diva(rates, empirical, predicted, flags):
    logger.info("Generating DIVA Detection Plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(rates, empirical, 'o-', label='Empirical Acc (Poisoned)')
    plt.plot(rates, predicted, 's--', label='Predicted Acc (Clean Model Expectation)')
    
    for i, flag in enumerate(flags):
        if flag:
            plt.axvspan(rates[i]-0.02, rates[i]+0.02, color='red', alpha=0.1, label='DIVA Detection' if i==0 else "")
            
    plt.title(f"DIVA Detection: Empirical vs Predicted Accuracy ({METHOD.upper()})")
    plt.xlabel("Poisoning Rate")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = f"./data/diva_{DATASET}_{METHOD}_plot.png"
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    logger.info("Starting Poisoning and Evaluation Pipeline")
    try:
        #step1_generate_poisoned_csvs()
        step2_evaluate_diva()
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.exception("A fatal error occurred during pipeline execution:")