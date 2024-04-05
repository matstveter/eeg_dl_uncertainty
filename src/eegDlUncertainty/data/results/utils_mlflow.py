from typing import Any, Dict, List, Optional, Tuple
import mlflow


def get_experiment_name(prediction_type: str, pairwise: List[str], one_class: str,
                        experiment_name: Optional[str] = None) -> None:
    """
    Determines and sets the name of the current machine learning experiment based on the type of prediction,
    and optionally, the specific classes involved in the prediction. It sets the experiment name in mlflow,
    ensuring that the experiment exists.

    Parameters
    ----------
    prediction_type : str
        The type of prediction for which the experiment is being run. This can be 'normal',
        'pairwise', or any other type leading to a 'one_class_vs_all' scenario.
    pairwise : tuple of str
        A tuple containing the names of the two classes involved in a pairwise comparison.
        This parameter is used only when prediction_type is 'pairwise'.
    one_class : str
        The name of the single class involved in a one-class versus all comparison. This parameter
        is used when prediction_type is not 'normal' or 'pairwise'.
    experiment_name : str, optional
        A custom name for the experiment. If provided, this name is used directly without
        deriving the name based on the prediction type or classes involved. Default is None.

    Side Effects
    ------------
    - Checks if the determined experiment name exists, and if not, it attempts to create it.
    - Sets the current experiment in mlflow to the determined or provided experiment name.

    Notes
    -----
    Ensure `mlflow` is properly configured and accessible within the environment where this
    function is used, as it directly interacts with mlflow's experiment tracking features.

    Examples
    --------
    >>> get_experiment_name("normal", ("class1", "class2"), "class1")
    This would set the experiment name to 'normal_vs_mci_vs_dementia', ensure it exists in mlflow,
    and set it as the current experiment.

    >>> get_experiment_name("pairwise", ("class1", "class2"), "class1")
    This would result in the experiment name being 'pairwise_class1_vs_class2', ensure it exists,
    and set it as the current experiment.

    """
    if experiment_name is None:
        if prediction_type == "normal":
            exp_name = "normal_vs_mci_vs_dementia"
        elif prediction_type == "pairwise":
            exp_name = f"{prediction_type}_{pairwise[0]}_vs_{pairwise[1]}"
        else:
            exp_name = f"{one_class}_vs_all"
    else:
        exp_name = experiment_name

    ensure_experiment_exists(exp_name)
    mlflow.set_experiment(exp_name)


def add_config_information(config: Dict[str, Any], dataset: str) -> None:
    """
    Logs configuration information and dataset name for the current MLflow experiment.

    This function iterates through a configuration dictionary to log various experiment parameters
    in MLflow. It handles special cases for logging 'pairwise' and 'one_vs_all' prediction types, including
    logging additional parameters that specify the classes involved in these prediction types. It ensures
    that configuration related to the prediction types is logged in a structured way.

    Parameters
    ----------
    config : dict
        A dictionary containing key-value pairs of configuration parameters for the experiment.
        Expected keys include 'prediction_type', 'pairwise_class', and 'which_one_vs_all_class'
        among potentially others. The function specially handles 'prediction_type' to log additional
        parameters based on its value.
    dataset : str
        The name of the dataset being used in the experiment. This is logged as a separate parameter.

    Notes
    -----
    - This function directly interacts with MLflow's logging API, and it assumes that an MLflow experiment
      has already been set up and started.
    - The function will skip logging 'which_one_vs_all_class' and 'pairwise_class' directly if they are not
      relevant to the 'prediction_type', to avoid cluttering the experiment's logged parameters with
      unnecessary information.
    """
    mlflow.log_param("Dataset", dataset)

    for key, val in config.items():
        if key == "prediction_type":
            if val == "pairwise":
                mlflow.log_param('pair', config['pairwise_class'])
            elif val == "one_vs_all":
                mlflow.log_param(key, val)
                mlflow.log_param("Focus Class", config["which_one_vs_all"])
            continue
        elif key == "which_one_vs_all" or key == "pairwise_class":
            continue
        else:
            mlflow.log_param(key, val)


def ensure_experiment_exists(experiment_name: str) -> str:
    """
    Ensure that an MLflow experiment exists. If the experiment does not exist, create it.
    Returns the experiment ID for the specified experiment name.

    Parameters:
    - experiment_name: The name of the experiment.

    Returns:
    - experiment_id: The ID of the existing or newly created experiment.
    """
    # Try to get the experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # If the experiment does not exist, create it and get its ID
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment created with ID: {experiment_id}")
    else:
        # If the experiment exists, use its ID
        experiment_id = experiment.experiment_id
        print(f"Experiment already exists with ID: {experiment_id}")

    return str(experiment_id)
