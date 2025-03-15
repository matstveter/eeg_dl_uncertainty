import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.CauDataGenerator import OODDataGenerator
from eegDlUncertainty.data.dataset.OODDataset import GreekEEGDataset, MPILemonDataset, TDBrainDataset
# from eegDlUncertainty.data.utils import save_dict_to_pickle
from eegDlUncertainty.data.file_utils import save_dict_to_pickle
from eegDlUncertainty.experiments.utils_exp import check_folder

FIG_SIZE = (20, 12)
TITLE_FONT = 30
LABEL_FONT = 28
TICK_FONT = 20
# Mapping from class indices to labels
class_labels = {0: "Normal", 1: "MCI", 2: "Dementia"}


def calculate_max_brier_score(num_classes):
    if num_classes == 1:
        return 0.0
    else:
        return 2.0


def create_scatter_plot(probs_pred, target_classes, dataset_name, save_path, jitter=0.15):
    """
    Creates and saves a scatter plot visualizing the Brier scores for predicted probabilities
    compared to the target classes, distinguishing between correct and incorrect predictions.

    Parameters
    ----------
    probs_pred : np.ndarray
        Array of predicted probabilities with shape (n_samples, n_classes).
    target_classes : np.ndarray
        One-hot encoded array of target classes with shape (n_samples, n_classes).
    dataset_name : str
        Name of the dataset to use in the plot title and saved filename.
    save_path : str
        Directory path where the plot will be saved.
    jitter : float, optional
        Amount of random jitter to add to the class labels for better visualization. Default is 0.15.

    Returns
    -------
    None
        The function saves the scatter plot as an EPS file in the specified directory and displays it.

    Notes
    -----
    The function calculates the Brier score for each sample, which is a measure of the accuracy of
    probabilistic predictions. The plot shows the Brier scores for each class, color-coded based on
    whether the predictions are correct or incorrect.
    """
    try:
        # Calculate the Brier score for each sample
        brier = np.mean((probs_pred - target_classes) ** 2, axis=1)
    except RuntimeWarning:
        print("runtime warning, ")
    # Get the predicted and true classes
    predictions = np.argmax(probs_pred, axis=1)
    true_classes = np.argmax(target_classes, axis=1)

    # Identify correct and incorrect predictions
    correct_predictions = predictions == true_classes
    # Create a DataFrame for Seaborn
    data = {
        'Class': true_classes + np.random.uniform(-jitter, jitter, size=true_classes.shape),
        'Brier Score': brier,
        'Correct': np.where(correct_predictions, 'Correct', 'Wrong')
    }
    df = pd.DataFrame(data)

    sns.set(style="darkgrid")

    plt.figure(figsize=FIG_SIZE)

    # Use Seaborn scatterplot
    sns.scatterplot(x='Class', y='Brier Score', hue='Correct', data=df,
                    palette={'Correct': 'green', 'Wrong': 'red'}, legend='full', s=125)
    # Draw a horizontal line for the maximum Brier score
    max_brier = calculate_max_brier_score(num_classes=probs_pred.shape[1])
    plt.axhline(y=max_brier, color='blue', linestyle='--', label='Max Brier Score')

    plt.xlabel("Class", fontsize=LABEL_FONT)

    # Dynamically set x-ticks based on present classes
    present_classes = np.unique(true_classes)
    present_class_labels = [class_labels[cls] for cls in present_classes]
    plt.xticks(ticks=present_classes, labels=present_class_labels, fontsize=TICK_FONT)

    plt.ylabel("Brier Score", fontsize=LABEL_FONT)
    plt.title(f'{dataset_name}: Performance vs. Uncertainty', fontsize=TITLE_FONT, weight='bold', color='navy')
    plt.legend(title='Prediction', fontsize=TICK_FONT, title_fontsize=LABEL_FONT)
    plt.tight_layout()

    plt.savefig(f"{save_path}/{dataset_name}_OOD.eps", dpi=300, format="eps")
    plt.close()


def get_loaders(dataset, device, batch_size):
    """
    This function creates a DataLoader object for a given dataset.

    Parameters:
    dataset (Dataset): The dataset for which the DataLoader is to be created.
    device (Device): The device on which the data loading operations are to be performed.
    batch_size (int): The number of samples per batch to load.

    Returns:
    DataLoader: A DataLoader object for the given dataset.
    """
    greek_data_gen = OODDataGenerator(dataset=dataset, use_age=True, device=device)
    return DataLoader(greek_data_gen, batch_size=batch_size, shuffle=False)


def get_dataset(dataset_version: int, num_seconds: int, age_scaling: str, device, batch_size: int,
                train_dataset):
    """
    This function creates DataLoader objects for three different datasets: GreekEEGDataset, MPILemonDataset, and
    TDBrainDataset.

    Parameters:
    dataset_version (int): The version of the dataset to be loaded.
    num_seconds (int): The number of seconds of EEG data to be loaded.
    age_scaling (str): The scaling method to be applied to the age data in the dataset.
    device (Device): The device on which the data loading operations are to be performed.
    batch_size (int): The number of samples per batch to load.
    train_dataset (CauEEGDataset): The training dataset to be used for age scaling.

    Returns:
    tuple: A tuple containing DataLoader objects for the GreekEEGDataset, MPILemonDataset, and TDBrainDataset.
    """
    ages = train_dataset.get_age_info()

    greek = GreekEEGDataset(dataset_version=dataset_version, num_seconds_eeg=num_seconds, age_scaling=age_scaling,
                            ages=ages)
    mpi = MPILemonDataset(dataset_version=dataset_version, num_seconds_eeg=num_seconds, age_scaling=age_scaling,
                          ages=ages)
    tdbrain = TDBrainDataset(dataset_version=dataset_version, num_seconds_eeg=num_seconds, age_scaling=age_scaling,
                             ages=ages)

    return (get_loaders(dataset=greek, device=device, batch_size=batch_size),
            get_loaders(dataset=mpi, device=device, batch_size=batch_size),
            get_loaders(dataset=tdbrain, device=device, batch_size=batch_size))


def single_dataset_experiment(ensemble_class, data_loader, device, dataset_name, save_path):
    """
    This function performs an experiment on a single dataset.

    Parameters:
    model (Model or list of Model): The model or ensemble of models to be used for the experiment.
    data_loader (DataLoader): The DataLoader object for the dataset to be used in the experiment.
    device (Device): The device on which the experiment is to be performed.

    Returns:
    dict: A dictionary containing the performance metrics, uncertainty metrics, and class-wise uncertainty metrics of
     the experiment.
    dict: A dictionary containing the mean logits, all probabilities, probabilities, predictions, and target classes of
    the experiment.
    """
    results = ensemble_class.ensemble_performance_and_uncertainty(data_loader, device, save_path,
                                                                  save_name=f"OOD_{dataset_name}",
                                                                  save_to_pickle=True, save_to_mlflow=True)

    predictions = results['average_epochs_merge_softmax']['predictions']
    create_scatter_plot(probs_pred=predictions["final_subject_probabilities"], 
                        target_classes=predictions["subject_one_hot_labels"],
                        dataset_name=dataset_name, save_path=save_path)

    print("Finished with dataset: ", dataset_name)
    return results, predictions


def apply_jitter_with_centered_point(df, jitter=0.15):
    """
    Apply jitter to all but one point per class in the dataframe to ensure one point remains at the class value.

    Parameters
    ----------
    df : pd.DataFrame
        containing 'Class' column.
    jitter : float
        Amount of random jitter to add to the class labels for better visualization.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with jitter applied to 'Class_Jittered' column.
    """
    jittered_classes = df['Class'].copy()
    for cls in df['Class'].unique():
        class_indices = df.index[df['Class'] == cls].tolist()
        if len(class_indices) > 1:
            jitter_indices = np.random.choice(class_indices, len(class_indices) - 1, replace=False)
            jittered_classes[jitter_indices] = jittered_classes[jitter_indices] + np.random.uniform(-jitter, jitter,
                                                                                                    size=len(
                                                                                                        jitter_indices))
    df['Class_Jittered'] = jittered_classes
    return df


def all_dataset_scatter_plots(probs_pred_list, target_classes_list, dataset_names, save_path, jitter=0.15):
    """
    Creates scatter plots for multiple datasets and combines them into one figure with subplots.

    Parameters
    ----------
    probs_pred_list : list of np.ndarray
        List of arrays with predicted probabilities, each with shape (n_samples, n_classes).
    target_classes_list : list of np.ndarray
        List of one-hot encoded arrays of target classes, each with shape (n_samples, n_classes).
    dataset_names : list of str
        List of dataset names.
    save_path : str
        Directory path where the plot will be saved.
    jitter : float, optional
        Amount of random jitter to add to the class labels for better visualization. Default is 0.15.

    Returns
    -------
    None
    """
    sns.set(style="darkgrid")

    num_datasets = len(probs_pred_list)
    fig, axes = plt.subplots(1, num_datasets, figsize=(9 * num_datasets, 10), sharey=True)

    for i, (probs_pred, target_classes, dataset_name) in enumerate(
            zip(probs_pred_list, target_classes_list, dataset_names)):
        # Calculate the Brier score for each sample
        brier = np.mean((probs_pred - target_classes) ** 2, axis=1)

        # Get the predicted and true classes
        predictions = np.argmax(probs_pred, axis=1)
        true_classes = np.argmax(target_classes, axis=1)

        # Identify correct and incorrect predictions
        correct_predictions = predictions == true_classes

        # Create a DataFrame for Seaborn
        data = {
            'Class': true_classes,
            'Brier Score': brier,
            'Correct': np.where(correct_predictions, 'Correct', 'Wrong')
        }
        df = pd.DataFrame(data)

        # Apply jitter to all but one point per class
        df = apply_jitter_with_centered_point(df, jitter)

        sns.scatterplot(x='Class_Jittered', y='Brier Score', hue='Correct', data=df, ax=axes[i],
                        palette={'Correct': 'green', 'Wrong': 'red'}, legend=True, s=125)
        max_brier = calculate_max_brier_score(num_classes=probs_pred.shape[1])
        axes[i].axhline(y=max_brier, color='blue', linestyle='--', label='Max Brier Score')
        axes[i].set_title(dataset_name, fontsize=TITLE_FONT / 2, weight='bold', color='navy')

        # Dynamically set x-ticks based on present classes
        present_classes = np.unique(true_classes)
        present_class_labels = [class_labels[cls] for cls in present_classes]

        axes[i].set_xticks(present_classes)
        axes[i].set_xticklabels(present_class_labels, fontsize=TICK_FONT)

        # Set x-limits to ensure jitter stays within bounds
        if len(present_classes) == 1:
            axes[i].set_xlim(present_classes[0] - jitter, present_classes[0] + jitter)
        else:
            axes[i].set_xlim(-0.5, len(class_labels) - 0.5)

        if i == 0:
            axes[i].set_ylabel("Brier Score", fontsize=LABEL_FONT)
            # Set y tickfont to be the same as x tickfont
            axes[i].tick_params(axis='y', labelsize=TICK_FONT)
        else:
            axes[i].set_ylabel('')

        axes[i].set_xlabel('')
        # Add legend values to the figure so that these can be plotted for the large figure
        axes[i].legend().set_visible(False)

    # Assuming `fig` is your matplotlib figure object
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]  # ignore [var-annotated]

    # Remove duplicates while preserving order
    unique_labels = []
    unique_lines = []
    for line, label in zip(lines, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_lines.append(line)

    fig.legend(unique_lines, unique_labels, fontsize=TICK_FONT, loc='upper right')

    # Set common labels and title
    fig.text(0.5, 0.04, 'Class', ha='center', va='center', fontsize=LABEL_FONT)
    fig.suptitle('Performance vs Uncertainty', fontsize=TITLE_FONT, weight='bold', color='navy', y=1.0)

    # plt.tight_layout() but not on the height
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{save_path}/OOD.eps", dpi=300, format="eps")
    plt.close()


def ood_exp(ensemble_class, dataset_version: int, num_seconds: int, age_scaling: str, device, batch_size: int,
            save_path: str, train_dataset):
    print("Running OOD experiment")
    greek_loader, mpi_loader, tdbrain_loader = get_dataset(dataset_version=dataset_version,
                                                           num_seconds=num_seconds,
                                                           device=device, batch_size=batch_size,
                                                           age_scaling=age_scaling,
                                                           train_dataset=train_dataset)

    save_path = check_folder(path=save_path, path_ext="figures")

    print("Running Militadous Experiment")
    greek_res, greek_pred = single_dataset_experiment(ensemble_class=ensemble_class,
                                                      data_loader=greek_loader, device=device,
                                                      dataset_name="Miltiadous", save_path=save_path)
    print("Running MPI Experiment")
    mpi_res, mpi_pred = single_dataset_experiment(ensemble_class=ensemble_class, data_loader=mpi_loader, device=device,
                                                  dataset_name="MPI", save_path=save_path)
    print("Running TDBrain Experiment")
    tdbrain_res, tdbrain_pred = single_dataset_experiment(ensemble_class=ensemble_class, data_loader=tdbrain_loader,
                                                          device=device,
                                                          dataset_name="TDBrain", save_path=save_path)
    combined_results = {'greek': greek_res, 'mpi': mpi_res, 'tdbrain': tdbrain_res}

    preds_prob_list = [greek_pred["final_subject_probabilities"], 
                       mpi_pred["final_subject_probabilities"], 
                       tdbrain_pred["final_subject_probabilities"]]
    targets_list = [greek_pred["subject_one_hot_labels"], 
                    mpi_pred["subject_one_hot_labels"], 
                    tdbrain_pred["subject_one_hot_labels"]]
    all_dataset_scatter_plots(probs_pred_list=preds_prob_list, target_classes_list=targets_list,
                              dataset_names=["Miltiadous", "MPI", "TDBrain"], save_path=save_path)

    save_dict_to_pickle(data_dict=combined_results, path=save_path, name="ood_results")
    return combined_results
