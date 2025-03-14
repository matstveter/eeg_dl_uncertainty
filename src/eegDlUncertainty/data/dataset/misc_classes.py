import os

import numpy as np
from sklearn.preprocessing import StandardScaler


class AgeScaler:
    def __init__(self, dataset_dict, scaling_type):
        self._dataset_dict = dataset_dict
        self._scaling_type = scaling_type

        age_list = []
        for k, v in dataset_dict.items():
            age_list.append(v['age'])
        ages = np.array(age_list)

        if scaling_type == "min_max":
            self._min = np.min(ages)
            self._max = np.max(ages)
        elif scaling_type == "sklearn_scale":
            ages_reshaped = ages.reshape(-1, 1)  # Reshape to 2D array for sklearn
            self._scaler = StandardScaler()
            self._scaler.fit(ages_reshaped)
        else:
            raise ValueError("Invalid scaling type. Choose from 'min_max', 'sklearn_scale'.")

    def transform(self, sub_ids):
        transformed_ages = []
        for sub in sub_ids:
            age = self._dataset_dict[sub]['age']
            if self._scaling_type == "min_max":
                scaled_age = (age - self._min) / (self._max - self._min)
            else:
                scaled_age = self._scaler.transform(np.array([[age]]))[0][0]  # Transform expects 2D input
            transformed_ages.append(scaled_age)
        return np.array(transformed_ages)
    
    def inverse_transform(self, scaled_ages):
        original_ages = []
        for scaled_age in scaled_ages:
            if self._scaling_type == "min_max":
                original_age = scaled_age * (self._max - self._min) + self._min
            else:
                original_age = self._scaler.inverse_transform(np.array([[scaled_age]]))[0][0]
            original_ages.append(original_age)
        return np.array(original_ages)


def verify_split_subjects(subject_list, path):
    sub_ids = [file_name.split(".")[0] for file_name in os.listdir(path)]

    verified_subjects = []
    for s in subject_list:
        if s in sub_ids:
            verified_subjects.append(s)
    return verified_subjects
