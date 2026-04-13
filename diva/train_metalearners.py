import os
import itertools
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from aim import Run

def setup_logger(output_dir):
    os.makedirs("./logs", exist_ok=True)
    logger = logging.getLogger("MetaLearner_Trainer")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    fh = logging.FileHandler("./logs/metalearner_training.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def load_meta_database(file_path, logger):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return pd.DataFrame()


def combine_meta_databases(file_paths, logger):
    """Combines multiple meta databases by concatenating rows."""
    combined_data = pd.DataFrame()
    for file_path in file_paths:
        data = load_meta_database(file_path, logger)
        if not data.empty:
            combined_data = pd.concat([combined_data, data], axis=0, ignore_index=True)
    return combined_data


def train_and_evaluate_svm(X, y, model_name, output_folder, logger, aim_run):
    """Trains the SVR Oracle and tracks performance with Aim."""
    # Ensure 2D array
    X_flattened = X.reshape((X.shape[0], -1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_flattened, y, test_size=0.2, random_state=42
    )

    # Train an SVM model
    svm_model = SVR()
    svm_model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = svm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    logger.info(f"Model Combination: [{model_name}] | MSE: {mse:.4f}")

    # --- AIM METRIC TRACKING ---
    aim_run.track(mse, name='mean_squared_error', context={'combination': model_name})

    # Save the model to the 'metalearners' folder
    model_filename = os.path.join(output_folder, f"metalearner_{model_name}.pkl")
    joblib.dump(svm_model, model_filename)


if __name__ == "__main__":
    # --- Argparse Setup ---
    parser = argparse.ArgumentParser(description="Train SVR Meta-Learners on MetaDB combinations")
    parser.add_argument("--input_dir", type=str, default="./data/metadbs", 
                        help="Folder containing all meta database CSVs")
    parser.add_argument("--output_dir", type=str, default="./data/metalearners", 
                        help="Output folder for trained models")
    parser.add_argument("--description", type=str, default="Training SVR Meta-Learner Oracles", 
                        help="Description of this run to be stored in Aim")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)

    # Get all the CSV files from the folder
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        exit(1)

    csv_files = [f for f in os.listdir(args.input_dir) if f.endswith(".csv")]
    
    if not csv_files:
        logger.warning(f"No CSV files found in {args.input_dir}. Exiting.")
        exit(0)

    logger.info(f"Found {len(csv_files)} MetaDB files. Starting combinatorics training...")

    # --- Initialize Aim Run ---
    run = Run(experiment="MetaLearner_Training")
    run.set("description", args.description)
    
    # Log Hyperparameters
    run["hparams"] = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "total_source_files": len(csv_files),
        "model_type": "SVR"
    }

    try:
        # Iterate through all possible combinations of the meta databases
        for r in range(1, len(csv_files) + 1):
            for combination in itertools.combinations(csv_files, r):
                
                # Full paths to the meta database files
                file_paths = [os.path.join(args.input_dir, file) for file in combination]

                # Load and combine the meta databases
                combined_data = combine_meta_databases(file_paths, logger)

                if combined_data.empty:
                    continue

                # Extract complexity measure columns ('c1' to 't4') and target
                # fillna ensures uniform feature dimensions even if some measures failed
                complexity_measures = combined_data.loc[:, "c1":"t4"].fillna(0.0).to_numpy() 
                target = combined_data["Test.Clean"].to_numpy()

                # Create a clean model name based on the combination
                model_name = "+".join(
                    [os.path.splitext(f)[0].replace("meta_database_", "") for f in combination]
                )

                # Train and evaluate the SVM model
                train_and_evaluate_svm(
                    X=complexity_measures, 
                    y=target, 
                    model_name=model_name, 
                    output_folder=args.output_dir, 
                    logger=logger, 
                    aim_run=run
                )
                
        logger.info("Meta-Learner combinatorics training completed successfully.")
        
    except Exception as e:
        logger.exception("A fatal error occurred during training:")
    finally:
        run.close()