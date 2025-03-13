import os
import pickle


def save_dict_to_pickle(data_dict, path, name):
    full_path = os.path.join(path, f"{name}.pkl")

    # Check if file exists, if so, try another name
    if os.path.exists(full_path):
        print("File already exists, trying another name.")
        i = 1
        while os.path.exists(full_path):
            full_path = os.path.join(path, f"{name}_{i}.pkl")
            i += 1

    with open(full_path, 'wb') as file:
        pickle.dump(data_dict, file)
