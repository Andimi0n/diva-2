import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import ParameterGrid

from scripts.svm_poissvm.svm_poissvm_generate_metadb import poissvm_poison
from scripts.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import (
    feature_noise_poison,
)
from scripts.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import (
    random_flip_poison,
)
from scripts.svm_alfa.svm_alfa_generate_metadb import alfa_poison


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


def extract_complexity_measures(input_path):
    # Find all poisoned datasets
    poisoned_files = glob.glob(os.path.join(input_path, "*.csv"))

    results = []
    for file in poisoned_files:
        print(f"Computing c-measures for: {file}...")
        # Load the poisoned dataset
        data = pd.read_csv(file)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Initialize MFE (meta feature extractor) with complexity measures
        mfe = MFE(groups=["complexity"])

        # Fit the MFE model to the data
        mfe.fit(X, y)

        # Extract meta-features
        features, values = mfe.extract()

        # Collect results
        result = {"file": os.path.basename(file)}
        result.update(dict(zip(features, values)))
        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df


def extract_key(filename):
    # Extract the part that matches the pattern in the file names
    filename = os.path.basename(filename)  # Get the file name without the path
    return "_".join(filename.split("_")[:10])  # Return the core file name part


def make_metadb(csv_path, cmeasure_dataframe, output_path):
    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        print(f"No CSV found at {csv_path}. Creating a new CSV at {output_path}.")
        cmeasure_dataframe.to_csv(csv_path, index=False)
        return

    # Read the CSV file
    csv_data = pd.read_csv(csv_path)

    # Extract the key from the 'Path.Poison' column for matching
    csv_data["key"] = csv_data["Path.Poison"].apply(extract_key)

    # Adjust the 'file' column in cmeasure_dataframe to match the format of 'key'
    cmeasure_dataframe["key"] = cmeasure_dataframe["file"].apply(extract_key)

    # Log data to debug
    print("CSV Data 'key' column:", csv_data["key"].head())
    print("DataFrame 'key' column:", cmeasure_dataframe["key"].head())

    # Merge the two datasets on the extracted key
    merged_data = pd.merge(csv_data, cmeasure_dataframe, on="key", how="inner")

    if merged_data.empty:
        print("No matching data found for merging. Check the 'key' and 'file' columns.")

    # Save the merged data to a new CSV file
    merged_data.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--nSets", default=10, type=int, help="# of random generated synthetic datasets."
    )
    parser.add_argument(
        "-f", "--folder", default="output", type=str, help="The output folder."
    )
    parser.add_argument(
        "-s", "--step", type=float, default=0.05, help="Spacing between poisoning rates. Default=0.05."
    )
    parser.add_argument(
        "-m", "--max", type=float, default=0.41, help="End of interval for poisoning rates. Default=0.41."
    )
    args = parser.parse_args()

  # Create the base folder
    base = args.folder
    os.makedirs(base, exist_ok=True)

    # Define poisoning methods with properly constructed paths
    # TODO: change that into class or something better, more robust than a dictionary
    poisoning_methods = {
        "alfa_svm": {
            "poison_function": alfa_poison,
            "complexity_dir": os.path.join(base, "poisoned_data", "alfa_svm"),
            "csv_score": os.path.join(base, "poisoned_data", "synth_alfa_svm_score.csv"),
            "meta_db": os.path.join(base, "meta_database_alfa_svm.csv"),
        },
        "feature_noise_svm": {
            "poison_function": feature_noise_poison,
            "complexity_dir": os.path.join(base, "poisoned_data", "feature_noise_svm"),
            "csv_score": os.path.join(base, "poisoned_data", "synth_feature_noise_svm_score.csv"),
            "meta_db": os.path.join(base, "meta_database_feature_noise_svm.csv"),
        },
        "random_flip_svm": {
            "poison_function": random_flip_poison,
            "complexity_dir": os.path.join(base, "poisoned_data", "random_flip_svm"),
            "csv_score": os.path.join(base, "poisoned_data", "synth_random_flip_svm_score.csv"),
            "meta_db": os.path.join(base, "meta_database_random_flip_svm.csv"),
        },
        "poissvm": {
            "poison_function": poissvm_poison,
            "complexity_dir": os.path.join(base, "poisoned_data", "numerical_gradient"),
            "csv_score": os.path.join(base, "poisoned_data", "synth_poissvm_score.csv"),
            "meta_db": os.path.join(base, "meta_database_poissvm_svm.csv"),
        },
    }


    advx_range = np.arange(0, args.max, args.step)

    for method_name, method_info in poisoning_methods.items():
        generated_files = generate_synthetic_data(args.nSets, args.folder)

        print(f"\nProcessing poisoning method: {method_name}")

        # Apply poisoning using the corresponding function
        #TODO: for poison_function, main class, each of them inherit from it to ensure proper parameters
        method_info["poison_function"](generated_files, advx_range, os.path.join(base, "poisoned_data"))

        # Step 3: Compute complexity measures from clean/poisoned files
        complexity_measures_df = extract_complexity_measures(
            method_info["complexity_dir"]
        )
        print(f"Complexity measures for {method_name}:")
        print(complexity_measures_df)

        # Step 4: Make meta-database from information gathered
        csv_path = method_info["csv_score"]
        meta_db_name = method_info["meta_db"]
        make_metadb(csv_path, complexity_measures_df, meta_db_name)

        print(f"Finished processing {method_name}")