import os
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
        # Train a default linear classifier if none is provided
        #if clf is None:
        #    clf = SVC(kernel='linear')
        #    clf.fit(X, y)
        #    print(f"      [ALFA] Clean baseline model score: {clf.score(X,y):.4f}")
            
        #from scripts.svm_alfa.utils.alfa import alfa
        from enron_benchmark.alfa import alfa_pytorch
        # Execute ALFA optimization
        y_poisoned = alfa_pytorch(X, y, rate, max_iter=5)
    
    elif method == 'alfa_poison':
        _, y_poisoned = alfa_poison(X, y, epsilon=rate)
        
    else:
        raise NotImplementedError(f"Poisoning method '{method}' is not implemented.")

    return X.copy(), y_poisoned

# --- Configuration ---
ENRON_NPZ_PATH = "./data/enron1_processed_sparse.npz"
MODEL_PATH = "./results/metalearners/metalearner_feature_noise_svm+random_flip_svm+poissvm_svm+alfa_svm.pkl" 
CSV_OUTPUT_DIR = "./data/enron_poisoned_csvs/"
SVD_MODEL_PATH = "./data/enron_svd_model.pkl" # NEW: Path to save SVD model
METHOD = 'alfa'
MAX_SAMPLE = 10000

def step1_generate_poisoned_csvs():
    print("--- STEP 1: Generating Poisoned CSVs ---")
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    
    f = np.load(ENRON_NPZ_PATH, allow_pickle=True, encoding='latin1')
    X_train_sparse = f['X_train'].reshape(1)[0]
    y_train = f['Y_train'] * 2 - 1

    # 1. Handle Dimensionality Reduction & Sparsity
    if METHOD == 'alfa':
        print("   -> Applying TruncatedSVD (n_components=100) for ALFA...")
        svd = TruncatedSVD(n_components=100, random_state=42)
        X_train_dense = svd.fit_transform(X_train_sparse) # Returns a dense array
        
        # Save SVD so the test set can be transformed in Step 2
        joblib.dump(svd, SVD_MODEL_PATH)
        
        # 2. Handle Downsampling
        if X_train_dense.shape[0] > MAX_SAMPLE:
            print(f"   -> Downsampling from {X_train_dense.shape[0]} to {MAX_SAMPLE} samples...")
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
        file_path = os.path.join(CSV_OUTPUT_DIR, f"enron_{METHOD}_noise_{rate:.2f}.csv")
        print(f"\nInjecting poisoning at rate {rate:.2f}...")
        
        X_noisy, y_train_pois = poison_data(X_train_dense, y_train, rate, method=METHOD)
        
        df = pd.DataFrame(X_noisy)
        df['target_label'] = y_train_pois
        df.to_csv(file_path, index=False)
        print(f"   -> Saved: {file_path}")

def step2_evaluate_diva():
    print("\n--- STEP 2: DIVA Evaluation ---")
    f = np.load(ENRON_NPZ_PATH, allow_pickle=True, encoding='latin1')
    X_test_sparse = f['X_test'].reshape(1)[0]
    y_test = f['Y_test'] * 2 - 1
    
    # NEW: Match Test Set dimensions to Train Set dimensions
    if METHOD == 'alfa' and os.path.exists(SVD_MODEL_PATH):
        print("   -> Loading SVD model to transform test data...")
        svd = joblib.load(SVD_MODEL_PATH)
        X_test_dense = svd.transform(X_test_sparse)
    else:
        X_test_dense = X_test_sparse.toarray() if sp.issparse(X_test_sparse) else X_test_sparse
    
    meta_learner = joblib.load(MODEL_PATH)
    advx_range = np.arange(0.0, 0.41, 0.05)
    
    results_empirical, results_predicted, results_flags = [], [], []
    
    for rate in advx_range:
        file_path = os.path.join(CSV_OUTPUT_DIR, f"enron_{METHOD}_noise_{rate:.2f}.csv")
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
        
        # Formatting the output to be cleaner
        na_count = C_measures.isna().sum().sum()
        print(f"Rate {rate:.2f} | Emp: {acc_emp:.4f} | Pred: {acc_pred:.4f} | Flag: {is_flagged} | Meta-NAs: {na_count}")

    visualize_diva(advx_range, results_empirical, results_predicted, results_flags)

def visualize_diva(rates, empirical, predicted, flags):
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
    plt.savefig(f"./data/diva_{METHOD}_plot.png")
    print(f"\n   -> Plot saved to ./data/diva_{METHOD}_plot.png")

if __name__ == "__main__":
    step1_generate_poisoned_csvs()
    step2_evaluate_diva()