import os
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sparse
from pymfe.mfe import MFE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# --- Configuration ---
ENRON_NPZ_PATH = "./data/enron1_processed_sparse.npz"
MODEL_PATH = "./results/metalearners/metalearner_feature_noise_svm+random_flip_svm+poissvm_svm+alfa_svm.pkl" 

import numpy as np

def poison_data_in_ram(X, y, rate, method='random', clf=None):
    """
    Poisons a dataset directly in RAM.
    
    Parameters:
    - X: Feature matrix (numpy array or scipy sparse matrix).
    - y: Label vector.
    - rate: Fraction of the dataset to poison (0.0 to 1.0).
    - method: 'random' or 'alfa'.
    - clf: Trained classifier (required for ALFA).
    
    Returns:
    - X_poisoned: The (potentially modified) feature matrix.
    - y_poisoned: The flipped label vector.
    """
    if rate == 0.0:
        return X.copy(), y.copy()

    X_poisoned = X.copy()
    y_poisoned = y.copy()

    if method == 'random':
        # Simple random label flipping
        n_flip = int(len(y) * rate)
        flip_indices = np.random.choice(len(y), n_flip, replace=False)
        
        # Flips assuming labels are strictly -1 and 1
        # If your labels are 0 and 1, use: y_poisoned[flip_indices] = 1 - y_poisoned[flip_indices]
        y_poisoned[flip_indices] = -y_poisoned[flip_indices]
        
    elif method == 'alfa':
        ALFA_MAX_ITER = 5  # Number of iterations for ALFA.
        N_ITER_SEARCH = 50  # Number of iterations for SVM parameter tuning.
        SVM_PARAM_DICT = {
            'C': loguniform(0.01, 10),
            'gamma': loguniform(0.01, 10),
            'kernel': ['rbf'],
        }
        clf = SVC()
        random_search = RandomizedSearchCV(
            clf,
            param_distributions=SVM_PARAM_DICT,
            n_iter=N_ITER_SEARCH,
            cv=5,
            n_jobs=-1,
        )
        random_search.fit(X, y)
        best_params = random_search.best_params_

        # Train model
        clf = SVC(**best_params)
        clf.fit(X, y)
        # ALFA adversarial label flipping
        if clf is None:
            raise ValueError("The 'alfa' method requires a trained classifier object (clf).")
            
        # Ensure these are imported at the top of your script
        from scripts.svm_alfa.utils.alfa import alfa
        from scripts.svm_alfa.utils.utils import transform_label
        
        # ALFA typically requires labels to be transformed prior to flipping
        #y_trans = transform_label(y_poisoned, target=-1)
        
        # Execute ALFA optimization in RAM
        y_poisoned = alfa(X_poisoned, y_poisoned, rate, svc_params=clf.get_params(), max_iter=5)
        
        
    else:
        raise NotImplementedError(f"Poisoning method '{method}' is not implemented for in-RAM processing.")

    return X_poisoned, y_poisoned

def test_enron_npz_pipeline():
    print("1. Loading and Preprocessing Enron .npz Data...")
    
    # Load the .npz file 
    f = np.load(ENRON_NPZ_PATH, allow_pickle=True, encoding='latin1')

    # Replicate the exact loading logic from the authors' datasets.py
    X_train = f['X_train'].reshape(1)[0]
    y_train = f['Y_train'] * 2 - 1
    
    X_test = f['X_test'].reshape(1)[0]
    y_test = f['Y_test'] * 2 - 1

    print(f"Loaded X_train: {X_train.shape[0]} samples, {X_train.shape[1]} features.")

    print("\n2. Evaluating Empirical Accuracy...")
    
    # Sklearn's SVC natively supports scipy.sparse matrices, so we pass it directly
    classifier = SVC(kernel='rbf')
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    acc_empirical = accuracy_score(y_test, y_pred)
    print(f"Honest Empirical Accuracy on Enron: {acc_empirical:.4f}")

    print("\n3. Extracting Complexity Measures (C-Measures)...")
    
    print("Subsampling data for MFE extraction to prevent memory exhaustion...")
    # Subsample directly from the sparse matrix
    X_sample_sparse, y_sample = resample(X_train, y_train, n_samples=1000, stratify=y_train, random_state=42)
    
    # PyMFE usually expects dense numpy arrays for its mathematical calculations.
    # Because we subsampled down to 1000 rows, converting to dense here won't crash your RAM.
    X_sample_dense = X_sample_sparse.toarray()
    
    mfe = MFE(groups=["complexity"])
    mfe.fit(X_sample_dense, y_sample) 
    features, values = mfe.extract()

    c_measures_df = pd.DataFrame([dict(zip(features, values))])
    
    X_meta_aligned = c_measures_df.fillna(0.0).to_numpy()
    X_flattened = X_meta_aligned.reshape(1, -1)

    print("\n4. Predicting Estimated Clean Accuracy...")
    meta_learner = joblib.load(MODEL_PATH)
    
    expected_dim = meta_learner.n_features_in_
    if X_flattened.shape[1] != expected_dim:
        print(f"CRITICAL WARNING: SVM expects {expected_dim} features but got {X_flattened.shape[1]}.")
        return

    acc_clean_estimated = meta_learner.predict(X_flattened)[0]
    print(f"Estimated Clean Accuracy: {acc_clean_estimated:.4f}")

    print("\n5. DIVA Assessment:")
    difference = abs(acc_empirical - acc_clean_estimated)
    print(f"Absolute Difference: {difference:.4f}")
    
    # Dynamic Threshold Heuristic from the paper: t = acc_empirical * (delta / 100)
    delta_tolerance = 5.0 # 5% tolerance
    threshold = acc_empirical * (delta_tolerance / 100)
    print(f"Threshold (assuming {delta_tolerance}% tolerance): {threshold:.4f}")
    
    if difference > threshold:
        print(">>> RESULT: FLAG AS POISONED")
    else:
        print(">>> RESULT: FLAG AS CLEAN")

if __name__ == "__main__":
    test_enron_npz_pipeline()