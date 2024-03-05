import os

from eegDlUncertainty.data.dataset_creator import create_eeg_dataset


def main():
    conf_name = 'data_processing.json'
    conf_path = os.path.join(os.path.dirname(__file__), conf_name)
    create_eeg_dataset(conf_path=conf_path)


if __name__ == "__main__":
    main()
