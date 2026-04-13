import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from scipy.stats import loguniform

from .utils.utils import create_dir, open_csv, to_csv
from ..base_poisoner import BasePoisoner

warnings.filterwarnings("ignore")

N_ITER_SEARCH = 50
SVM_PARAM_DICT = {
    "C": loguniform(0.01, 10),
    "gamma": loguniform(0.01, 10),
    "kernel": ["rbf"],
}

class RandomFlipPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="random_flip_svm", base_folder=base_folder)

    def random_label_flip(self, y_train, rate):
        y_flip = y_train.copy()
        n_flip = int(len(y_train) * rate)
        flip_indices = np.random.choice(len(y_train), size=n_flip, replace=False)
        y_flip[flip_indices] = 1 - y_flip[flip_indices]
        return y_flip

    def compute_and_save_flipped_data(self, X_train, y_train, X_test, y_test, clf, path_output_base, cols, flip_rate_range):
        acc_train_clean = clf.score(X_train, y_train)
        acc_test_clean = clf.score(X_test, y_test)

        accuracy_train_clean = [acc_train_clean] * len(flip_rate_range)
        accuracy_test_clean = [acc_test_clean] * len(flip_rate_range)
        accuracy_train_poison, accuracy_test_poison, path_poison_data_list = [], [], []

        for rate in flip_rate_range:
            path_poison_data = "{}_randomlabelflip_svm_{:.2f}.csv".format(path_output_base, np.round(rate, 2))
            try:
                if os.path.exists(path_poison_data):
                    X_train, y_flip, _ = open_csv(path_poison_data)
                else:
                    y_flip = self.random_label_flip(y_train, rate)
                    to_csv(X_train, y_flip, cols, path_poison_data)
                
                svm_params = clf.get_params()
                clf_poison = SVC(**svm_params)
                clf_poison.fit(X_train, y_flip)
                acc_train_poison = clf_poison.score(X_train, y_flip)
                acc_test_poison = clf_poison.score(X_test, y_test)
            except Exception as e:
                self.logger.error(e)
                acc_train_poison, acc_test_poison = 0, 0
            
            self.logger.info("     Flip Rate [{:.2f}]% - Acc  Poisoned Train: {:.2f}%  Test Set: {:.2f}%".format(
                rate * 100, acc_train_poison * 100, acc_test_poison * 100))
            
            path_poison_data_list.append(path_poison_data)
            accuracy_train_poison.append(acc_train_poison)
            accuracy_test_poison.append(acc_test_poison)

        return accuracy_train_clean, accuracy_test_clean, accuracy_train_poison, accuracy_test_poison, path_poison_data_list

    def apply_poisoning(self, file_paths, advx_range):
        for file_path in file_paths:
            self.logger.info(f"Started poisoning for {file_path}")
            X_train, y_train, cols = open_csv(file_path)
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
            dataname = Path(file_path).stem

            clf = SVC()
            random_search = RandomizedSearchCV(clf, param_distributions=SVM_PARAM_DICT, n_iter=N_ITER_SEARCH, cv=5, n_jobs=-1)
            random_search.fit(X_train, y_train)
            
            clf = SVC(**random_search.best_params_)
            clf.fit(X_train, y_train)

            acc_train_clean, acc_test_clean, acc_train_poison, acc_test_poison, path_poison_data_list = self.compute_and_save_flipped_data(
                X_train, y_train, X_test, y_test, clf,
                os.path.join(self.complexity_dir, dataname),
                cols, advx_range
            )

            data = {
                "Data": np.tile(dataname, reps=len(advx_range)),
                "Path.Poison": path_poison_data_list,
                "Rate": advx_range,
                "Train.Clean": acc_train_clean,
                "Test.Clean": acc_test_clean,
                "Train.Poison": acc_train_poison,
                "Test.Poison": acc_test_poison,
            }
            df = pd.DataFrame(data)
            df.to_csv(self.csv_score, mode='a' if os.path.exists(self.csv_score) else 'w', 
                      header=not os.path.exists(self.csv_score), index=False)