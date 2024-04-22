import numpy as np
from sklearn.preprocessing import StandardScaler


class AgeScaler:
    def __init__(self, dataset_dict, scaling_type):
        self._dataset_dict = dataset_dict
        self._scaling_type = scaling_type

        ages = []
        for k, v in dataset_dict.items():
            ages.append(v['age'])
        ages = np.array(ages)

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

                if add_noise:
                    # Inject Gaussian noise
                    noise = np.random.normal(0, noise_level * scaled_age)
                    scaled_age += noise

            transformed_ages.append(scaled_age)
        return np.array(transformed_ages)
