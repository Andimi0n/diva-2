import os
import glob
import pandas as pd
from abc import ABC, abstractmethod
from pymfe.mfe import MFE
import logging
from tqdm import tqdm

class BasePoisoner(ABC):
    """
    Abstract Base Class for all poisoning methods.
    Handles standard directory setup, complexity extraction, and MetaDB merging.
    """
    def __init__(self, name, base_folder, custom_complexity_dir=None):
        self.logger = logging.getLogger(name)
        self.name = name
        self.base_folder = base_folder
        
        # Setup standardized paths
        if custom_complexity_dir:
            self.complexity_dir = os.path.join(base_folder, "poisoned_data", custom_complexity_dir)
        else:
            self.complexity_dir = os.path.join(base_folder, "poisoned_data", name)
            
        self.csv_score = os.path.join(base_folder, "poisoned_data", f"synth_{name}_score.csv")
        self.meta_db = os.path.join(base_folder, "metadbs", f"meta_database_{name}.csv")

    @abstractmethod
    def apply_poisoning(self, file_path, advx_range):
        """
        MUST be implemented by child classes. 
        Contains the specific logic to poison the data and save the CSVs.
        """
        pass

    def extract_key(self, filename):
        """
        Extracts the core file name. Can be overridden by child classes if needed.
        """
        filename = os.path.basename(filename)
        return "_".join(filename.split("_")[:10])

    def extract_complexity_measures(self):
        """
        Shared logic to extract MFE complexity measures from the poisoned folder.
        """
        poisoned_files = glob.glob(os.path.join(self.complexity_dir, "*.csv"))

        output_path = os.path.join(self.complexity_dir, "complexity_measures.csv")
        if output_path in poisoned_files:
            poisoned_files.remove(output_path)
        
        results = []
        pbar = tqdm(poisoned_files, desc="Processing files")
        for file in pbar:
            pbar.set_postfix({'File': os.path.basename(file)})
            data = pd.read_csv(file)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values

            mfe = MFE(groups=["complexity"])
            mfe.fit(X, y)
            features, values = mfe.extract()

            result = {"file": os.path.basename(file)}
            result.update(dict(zip(features, values)))
            results.append(result)
        
        output_path = os.path.join(self.complexity_dir, "complexity_measures.csv")
        df_complexity = pd.DataFrame(results)
        df_complexity.to_csv(output_path, index=False)
        self.logger.info(f"Successfully saved complexity measures to {output_path}")

        return df_complexity
    
    def get_complexity_measures(self):
        """
        Retrieves complexity measures by loading a cached CSV.
        """
        output_path = os.path.join(self.complexity_dir, "complexity_measures.csv")

        # 1. Check if we have already processed these measures
        if os.path.exists(output_path):
            self.logger.info(f"Loading existing complexity measures from {output_path}")
            return pd.read_csv(output_path)
        else:
            raise ValueError("No complexity measures found, start by computing them.")

    def make_metadb(self, cmeasure_dataframe):
        """
        Shared logic to merge SVM scores with Complexity Measures.
        """
        if not os.path.exists(self.csv_score):
            self.logger.error(f"No CSV found at {self.csv_score}. Saving complexity data only.")
            cmeasure_dataframe.to_csv(self.meta_db, index=False)
            return

        csv_data = pd.read_csv(self.csv_score)
        
        # Apply key extraction
        csv_data["key"] = csv_data["Path.Poison"].apply(self.extract_key)
        cmeasure_dataframe["key"] = cmeasure_dataframe["file"].apply(self.extract_key)

        merged_data = pd.merge(csv_data, cmeasure_dataframe, on="key", how="inner")

        # Drop duplicates (Required by poissvm, safe for others)
        if "Path.Poison" in merged_data.columns:
            merged_data = merged_data.drop_duplicates(subset=["Path.Poison"])

        if merged_data.empty:
            self.logger.warning(f"Warning: No matching data found for merging in {self.name}.")

        merged_data.to_csv(self.meta_db, index=False)
        self.logger.info(f"Merged MetaDB saved to {self.meta_db}")

    def run_pipeline(self, file_paths, advx_range, entrypoint="poison"):
        """
        Executes the full pipeline for this specific poisoner.
        """
        self.logger.info(f"Starting Pipeline: {self.name.upper()}")
        os.makedirs(self.complexity_dir, exist_ok=True)
        cmeasure_df = None
        
        # 1. Apply Poisoning
        if entrypoint == "poison" :
            for (i,file) in enumerate(file_paths):
                self.logger.info(f"{i}/{len(file_paths)}: Started poisoning for file {file}")
                self.apply_poisoning(file, advx_range)

        if entrypoint in ("cmeasure", "poison") :
            # 2. Extract Complexity
            self.logger.info(f"\nExtracting complexity measures for {self.name}...")
            cmeasure_df = self.extract_complexity_measures()

        if entrypoint in ("metadb", "cmeasure", "poison") :
            if cmeasure_df is None:
                cmeasure_df = self.get_complexity_measures()

            # 3. Create MetaDB
            self.logger.info(f"\nCreating Meta Database for {self.name}...")
            self.make_metadb(cmeasure_df)