import argparse
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

# Import the base class
from scripts.base_poisoner import BasePoisoner

# Import your refactored specific poisoner classes
from scripts.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import FeatureNoisePoisoner
from scripts.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner
import logging


def generate_synthetic_data(n_sets, folder, mode='high_dim'):
    """
    Generates synthetic data with increased complexity to mimic 
    high-dimensional datasets like Enron or IMDB.
    """
    # Fix: N_SAMPLES now generates a range from 500 to 5000 in steps of 500
    N_SAMPLES_OPTIONS = np.arange(100, 1001, 100) 
    N_CLASSES = 2 

    data_path = os.path.join(folder, "clean_data")
    os.makedirs(data_path, exist_ok=True)

    # Realistic Feature Ranges
    # We include low-dim (4-100) and high-dim (100-1500) sets
    feature_ranges = list(range(10, 101, 20)) + [200, 500, 1000, 1500]
    
    grid = [] 
    for f in feature_ranges:
        # For high-dim text data, informative features are usually a small fraction
        informative_ratio = [0.05, 0.1, 0.2] if f > 100 else [0.5, 0.7]
        
        grid.append({
            "n_samples": N_SAMPLES_OPTIONS,
            "n_classes": [N_CLASSES],
            "n_features": [f],
            "n_repeated": [0, int(f * 0.1)], # 10% repeated features
            "n_informative": [max(2, int(f * r)) for r in informative_ratio],
            "weights": [[0.4, 0.6], [0.5, 0.5]], # Class balance
        })

    param_sets = list(ParameterGrid(grid))
    
    # Randomly shuffle and select indices
    replace = len(param_sets) < n_sets
    selected_indices = np.random.choice(len(param_sets), n_sets, replace=replace)

    generated_files = []

    for idx, i in enumerate(selected_indices):
        params = param_sets[i].copy()
        
        # Add Redundant Features: Mimic highly correlated word features
        # Redundant = total - informative - repeated
        remaining_space = params["n_features"] - params["n_informative"] - params["n_repeated"]
        params["n_redundant"] = np.random.randint(0, max(1, remaining_space))
        
        # Clusters per class: Higher clusters = more complex decision boundary
        params["n_clusters_per_class"] = np.random.randint(1, 4)
        params["random_state"] = np.random.randint(1000, 99999)

        # Generate the data
        X, y = make_classification(**params)
        
        # Scale data (Standard practice for SVM meta-learning)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        feature_names = [f"x{j}" for j in range(1, X.shape[1] + 1)]
        df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
        # Shift to {-1, 1} if needed by your ALFA implementation, otherwise keep 0,1
        df["y"] = y.astype(np.int32) 

        # Descriptive Filename
        file_name = "f{:04d}_i{:03d}_r{:03d}_n{:04d}".format(
            params["n_features"],
            params["n_informative"],
            params["n_redundant"],
            params["n_samples"]
        )

        data_list = glob.glob(os.path.join(data_path, f"{file_name}_*.csv"))
        postfix = str(len(data_list) + 1)

        output_path = os.path.join(data_path, f"{file_name}_{postfix}.csv")
        df.to_csv(output_path, index=False)
        
        if (idx + 1) % 5 == 0:
            logger.info(f"Generated {idx + 1}/{n_sets} files...")

        generated_files.append(output_path)

    return generated_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Main")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nSets", default=10, type=int, help="# of random generated synthetic datasets.")
    parser.add_argument("-f", "--folder", default="data", type=str, help="The output folder.")
    parser.add_argument("-s", "--step", type=float, default=0.05, help="Spacing between poisoning rates.")
    parser.add_argument("-m", "--max", type=float, default=0.41, help="End of interval for poisoning rates.")
    args = parser.parse_args()

    base = args.folder
    os.makedirs(base, exist_ok=True)
    advx_range = np.arange(0, args.max, args.step)

    # Generate synthetic datasets once
    logger.info("Generating synthetic datasets...")
    generated_files = generate_synthetic_data(args.nSets, args.folder)

    # Initialize all your poisoners
    poisoners = [
        AlfaPoisoner(base_folder=base),
        FeatureNoisePoisoner(base_folder=base),
        RandomFlipPoisoner(base_folder=base),
        PoisSVMPoisoner(base_folder=base)
    ]

    # Run the standardized pipeline for each method
    for poisoner in poisoners:
        poisoner.run_pipeline(generated_files, advx_range)