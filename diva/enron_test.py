import os
import numpy as np
import pandas as pd
import joblib
from pymfe.mfe import MFE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from enron_benchmark.alfa import alfa_poison

def poison_data(X, y, rate, method='random', clf=None):
    """
    Poisons a dataset directly in RAM.
    
    Parameters:
    - X: Feature matrix (numpy array or scipy sparse matrix).
    - y: Label vector.
    - rate: Fraction of the dataset to poison (0.0 to 1.0).
    - method: 'random', 'alfa', or 'feature_noise'.
    - clf: Trained classifier (required for ALFA).
    """
    if rate == 0.0:
        return X.copy(), y.copy()

    X_poisoned = X.copy()
    y_poisoned = y.copy()

    if method == 'random':
        # Simple random label flipping
        n_flip = int(len(y) * rate)
        flip_indices = np.random.choice(len(y), n_flip, replace=False)
        y_poisoned[flip_indices] = -y_poisoned[flip_indices]
        
    elif method == 'alfa':
        clf = SVC(kernel='rbf')
        clf.fit(X, y)
        # ALFA adversarial label flipping
        if clf is None:
            raise ValueError("The 'alfa' method requires a trained classifier object (clf).")
            
        from scripts.svm_alfa.utils.alfa import alfa
        # Execute ALFA optimization in RAM
        y_poisoned = alfa(X_poisoned, y_poisoned, rate, svc_params=clf.get_params(), max_iter=5)
    
    elif method == 'alfa_poison':
        X_poisoned, y_poisoned = alfa_poison(X,y, epsilon=rate)
        
    else:
        raise NotImplementedError(f"Poisoning method '{method}' is not implemented for in-RAM processing.")

    return X_poisoned, y_poisoned

# --- Configuration ---
ENRON_NPZ_PATH = "./data/enron1_processed_sparse.npz"
MODEL_PATH = "./results/metalearners/metalearner_feature_noise_svm+random_flip_svm+poissvm_svm+alfa_svm.pkl" 
CSV_OUTPUT_DIR = "./data/enron_poisoned_csvs/"

def step1_generate_poisoned_csvs():
    print("--- STEP 1: Generating Feature-Noise Poisoned CSVs ---")
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    
    f = np.load(ENRON_NPZ_PATH, allow_pickle=True, encoding='latin1')
    X_train_sparse = f['X_train'].reshape(1)[0]
    y_train = f['Y_train'] * 2 - 1 
    
    # Feature noise requires dense operations
    X_train_dense = X_train_sparse.toarray()
    advx_range = np.arange(0.0, 0.41, 0.05)

    #print("Fitting the SVC on the clean data")
    #clf = SVC(kernel='rbf')
    #clf.fit(X_train_dense, y_train)
    
    for rate in advx_range:
        file_path = os.path.join(CSV_OUTPUT_DIR, f"enron_noise_{rate:.2f}.csv")
        print(f"Injecting noise at rate {rate:.2f}...")
        
        # Use the alfa method
        X_noisy, y_train_copy = poison_data(X_train_dense, y_train, rate, method='alfa_poison')
        
        df = pd.DataFrame(X_noisy)
        df['target_label'] = y_train_copy
        df.to_csv(file_path, index=False)

def step2_evaluate_diva():
    print("\n--- STEP 2: DIVA Evaluation ---")
    f = np.load(ENRON_NPZ_PATH, allow_pickle=True, encoding='latin1')
    X_test_sparse = f['X_test'].reshape(1)[0]
    y_test = f['Y_test'] * 2 - 1
    X_test_dense = X_test_sparse.toarray()
    
    meta_learner = joblib.load(MODEL_PATH)
    advx_range = np.arange(0.0, 0.41, 0.05)
    
    results_empirical, results_predicted, results_flags = [], [], []
    
    for rate in advx_range:
        file_path = os.path.join(CSV_OUTPUT_DIR, f"enron_noise_{rate:.2f}.csv")
        df = pd.read_csv(file_path)
        y_train_pois = df['target_label'].values
        X_train_pois = df.drop('target_label', axis=1).values
        
        # Train and Evaluate
        clf = SVC(kernel='rbf')
        clf.fit(X_train_pois, y_train_pois)
        acc_emp = accuracy_score(y_test, clf.predict(X_test_dense))
        
        # Meta-feature extraction
        # Too heavy to calculate C-Measures on whole dataset -> only on 1000 points with same class balance as original dataset
        X_sample, y_sample = resample(X_train_pois, y_train_pois, n_samples=1000, stratify=y_train_pois)
        mfe = MFE(groups=["complexity"], suppress_warnings=True)
        mfe.fit(X_sample, y_sample)
        ft, vals = mfe.extract()
        C_measures = pd.DataFrame([dict(zip(ft, vals))])
        print(f"Number of na values: {C_measures.isna().sum()}")

        X_meta = C_measures.fillna(0.0).values
        acc_pred = meta_learner.predict(X_meta)[0]
        
        # DIVA Flagging Logic
        is_flagged = abs(acc_emp - acc_pred) > (acc_emp * 0.05)
        
        results_empirical.append(acc_emp)
        results_predicted.append(acc_pred)
        results_flags.append(is_flagged)
        print(f"Rate {rate:.2f} | Emp: {acc_emp:.4f} | Pred: {acc_pred:.4f} | Flag: {is_flagged}")

    visualize_diva(advx_range, results_empirical, results_predicted, results_flags)

def visualize_diva(rates, empirical, predicted, flags):
    plt.figure(figsize=(10, 6))
    plt.plot(rates, empirical, 'o-', label='Empirical Acc (Poisoned)')
    plt.plot(rates, predicted, 's--', label='Predicted Acc (Clean)')
    
    for i, flag in enumerate(flags):
        if flag:
            plt.axvspan(rates[i]-0.02, rates[i]+0.02, color='red', alpha=0.1, label='DIVA Detection' if i==0 else "")
            
    plt.title("DIVA Detection: Empirical vs Predicted Accuracy (Feature Noise)")
    plt.xlabel("Noise Injection Rate")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("./data/diva_feature_noise_plot.png")

if __name__ == "__main__":
    step1_generate_poisoned_csvs()
    step2_evaluate_diva()