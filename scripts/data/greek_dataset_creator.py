import os

from eegDlUncertainty.data.dataset.dataset_creator import create_eeg_dataset


def main():
    conf_name = 'data_processing_greek_eeg.json'
    conf_path = os.path.join(os.path.dirname(__file__), conf_name)
    create_eeg_dataset(conf_path=conf_path)


if __name__ == "__main__":
    main()
