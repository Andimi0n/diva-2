import os
import glob
import pandas as pd
from abc import ABC, abstractmethod
from pymfe.mfe import MFE

class BasePoisoner(ABC):
    """
    Abstract Base Class for all poisoning methods.
    Handles standard directory setup, complexity extraction, and MetaDB merging.
    """
    def __init__(self, name, base_folder, custom_complexity_dir=None):
        self.name = name
        self.base_folder = base_folder
        
        # Setup standardized paths
        if custom_complexity_dir:
            self.complexity_dir = os.path.join(base_folder, "poisoned_data", custom_complexity_dir)
        else:
            self.complexity_dir = os.path.join(base_folder, "poisoned_data", name)
            
        self.csv_score = os.path.join(base_folder, "poisoned_data", f"synth_{name}_score.csv")
        self.meta_db = os.path.join(base_folder, f"meta_database_{name}.csv")

    @abstractmethod
    def apply_poisoning(self, file_paths, advx_range):
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
        results = []
        for file in poisoned_files:
            data = pd.read_csv(file)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values

            mfe = MFE(groups=["complexity"])
            mfe.fit(X, y)
            features, values = mfe.extract()

            result = {"file": os.path.basename(file)}
            result.update(dict(zip(features, values)))
            results.append(result)

        return pd.DataFrame(results)

    def make_metadb(self, cmeasure_dataframe):
        """
        Shared logic to merge SVM scores with Complexity Measures.
        """
        if not os.path.exists(self.csv_score):
            print(f"No CSV found at {self.csv_score}. Saving complexity data only.")
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
            print(f"Warning: No matching data found for merging in {self.name}.")

        merged_data.to_csv(self.meta_db, index=False)
        print(f"Merged MetaDB saved to {self.meta_db}")

    def run_pipeline(self, file_paths, advx_range):
        """
        Executes the full pipeline for this specific poisoner.
        """
        print(f"\n{'='*50}\nExecuting Pipeline: {self.name.upper()}\n{'='*50}")
        os.makedirs(self.complexity_dir, exist_ok=True)
        
        # 1. Apply Poisoning
        self.apply_poisoning(file_paths, advx_range)
        
        # 2. Extract Complexity
        print(f"\nExtracting complexity measures for {self.name}...")
        cmeasure_df = self.extract_complexity_measures()
        
        # 3. Create MetaDB
        print(f"\nCreating Meta Database for {self.name}...")
        self.make_metadb(cmeasure_df)