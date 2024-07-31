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

        self._no_transformation = False
        if scaling_type == "min_max":
            self._min = np.min(ages)
            self._max = np.max(ages)
        elif scaling_type == "sklearn_scale":
            self._scaler = StandardScaler()
            self._scaler.fit(ages)
        elif scaling_type == "standard":  # Standard scaling as default
            self._mean = np.mean(ages)
            self._std = np.std(ages)
        else:
            self._no_transformation = True

    def transform(self, sub_ids, add_noise=False, noise_level=0.05):
        transformed_ages = []
        for sub in sub_ids:
            age = self._dataset_dict[sub]['age']
            if self._no_transformation:
                scaled_age = age
            else:
                if self._scaling_type == "min_max":
                    scaled_age = (age - self._min) / (self._max - self._min)
                elif self._scaling_type == "sklearn_scale":
                    scaled_age = self._scaler.transform(age)
                else:
                    scaled_age = (age - self._mean) / self._std

                if add_noise and self._scaling_type == "min_max":
                    # Inject Gaussian noise
                    noise = np.random.normal(0, noise_level * scaled_age)
                    scaled_age += noise

            transformed_ages.append(scaled_age)
        return np.array(transformed_ages)


def verify_split_subjects(subject_list, path):
    sub_ids = [file_name.split(".")[0] for file_name in os.listdir(path)]

    verified_subjects = []
    for s in subject_list:
        if s in sub_ids:
            verified_subjects.append(s)
    return verified_subjects
