import pytest

from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset


def test_valid_label_type():
    """Test that the class initializes correctly with a valid label type."""
    valid_label_type = "dementia"
    dataset_version = "1"
    # Assuming your class and methods don't need actual files to test this functionality
    dataset = CauEEGDataset(dataset_version, valid_label_type, eeg_len_seconds=60)
    assert dataset is not None, "Failed to initialize with valid label type"


def test_invalid_label_type():
    """Test that the class raises KeyError for an invalid label type."""
    invalid_label_type = "invalid_label"
    dataset_version = "1"
    with pytest.raises(KeyError):
        CauEEGDataset(dataset_version, invalid_label_type, eeg_len_seconds=60)


def test_nonexistent_dataset_version():
    """Test that the class raises an exception for a non-existent dataset version."""
    nonexistent_version = "999"  # Assuming this version doesn't exist
    valid_label_type = "dementia"
    with pytest.raises(KeyError):
        CauEEGDataset(nonexistent_version, valid_label_type, eeg_len_seconds=60)
