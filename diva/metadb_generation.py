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


def generate_synthetic_data(n_sets, folder):
    N_SAMPLES = np.arange(100, 200, 200) #! N_SAMPLES= [100] here. Not sure if it is a bug?
    N_CLASSES = 2  # Number of classes

    # Create directory
    data_path = os.path.join(folder, "clean_data")
    if not os.path.exists(data_path):
        print("Create path:", data_path)
        path = Path(data_path)
        path.mkdir(parents=True)

    grid = [] #Contains parameters used for scikit-learn dataset generator
    for f in range(4, 31):
        grid.append(
            {
                "n_samples": N_SAMPLES,
                "n_classes": [N_CLASSES],
                "n_features": [f],
                "n_repeated": [0],
                "n_informative": np.arange(f // 2, f + 1),
                "weights": [[0.4], [0.5], [0.6]],
            }
        )

    param_sets = list(ParameterGrid(grid))
    print("# of parameter sets:", len(param_sets))

    # Adjust redundant features and clusters per class for each parameter set
    for i in range(len(param_sets)):
        param_sets[i]["n_redundant"] = np.random.randint(
            0, high=param_sets[i]["n_features"] + 1 - param_sets[i]["n_informative"]
        )
        param_sets[i]["n_clusters_per_class"] = np.random.randint(
            1, param_sets[i]["n_informative"]
        )

    replace = len(param_sets) < n_sets
    selected_indices = np.random.choice(len(param_sets), n_sets, replace=replace)

    generated_files = []  # Keep track of generated files

    for i in selected_indices:
        param_sets[i]["random_state"] = np.random.randint(1000, np.iinfo(np.int16).max)

        X, y = make_classification(**param_sets[i])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        feature_names = ["x" + str(j) for j in range(1, X.shape[1] + 1)]
        df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
        df["y"] = y.astype(np.int32)  # Convert y to integers

        file_name = "f{:02d}_i{:02d}_r{:02d}_c{:02d}_w{:.0f}_n{}".format(
            param_sets[i]["n_features"],
            param_sets[i]["n_informative"],
            param_sets[i]["n_redundant"],
            param_sets[i]["n_clusters_per_class"],
            param_sets[i]["weights"][0] * 10,
            param_sets[i]["n_samples"],
        )

        data_list = glob.glob(os.path.join(data_path, file_name + "*.csv"))
        postfix = str(len(data_list) + 1)

        output_path = os.path.join(data_path, f"{file_name}_{postfix}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

        generated_files.append(output_path)  # Store the path of each generated file

    return generated_files  # Return the list of generated files for further processing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nSets", default=10, type=int, help="# of random generated synthetic datasets.")
    parser.add_argument("-f", "--folder", default="output", type=str, help="The output folder.")
    parser.add_argument("-s", "--step", type=float, default=0.05, help="Spacing between poisoning rates.")
    parser.add_argument("-m", "--max", type=float, default=0.41, help="End of interval for poisoning rates.")
    args = parser.parse_args()

    base = args.folder
    os.makedirs(base, exist_ok=True)
    advx_range = np.arange(0, args.max, args.step)

    # Generate synthetic datasets once
    print("Generating synthetic datasets...")
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