import matplotlib.pyplot as plt
import mlflow
import seaborn as sns

from eegDlUncertainty.experiments.utils_exp import check_folder

FIG_SIZE = (20, 12)
TITLE_FONT = 30
LABEL_FONT = 28
TICK_FONT = 20


class Plotter:

    def __init__(self, train_dict, val_dict, test_dict, test_dict_best_model, save_path=None):
        self._train_dict = train_dict
        self._val_dict = val_dict
        self._test_dict = test_dict
        self._test_dict_best_model = test_dict_best_model

        if save_path is not None:
            self._save_path = check_folder(path=save_path, path_ext="figures")
        else:
            self._save_path = save_path

        self.fig_size = (20, 12)
        self.dpi = 300
        self.title_font = 20
        self.tick_font = 16

    def produce_plots(self):
        self._plot_loss()
        self._plot_accuracy()
        self._plot_auc()
        self._plot_mcc()
        self._print_test_results()

    def _print_test_results(self):
        filename = f"{self._save_path}/test_results.txt"
        with open(filename, "w") as file:
            # First section
            for key, val in self._test_dict.items():
                val_str = val[0] if isinstance(val, list) and len(val) > 0 else val
                file.write(f"{key.upper():<15}: {val_str}\n")

            file.write("\nBest Model:\n")
            # Best Model section
            for key, val in self._test_dict_best_model.items():
                val_str = val[0] if isinstance(val, list) and len(val) > 0 else val
                file.write(f"{key.upper():<15}: {val_str}\n")

        mlflow.log_artifact(filename)

    def _plot_loss(self):
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        plt.plot(self._train_dict['loss'], label='Train Loss')
        plt.plot(self._val_dict['loss'], label='Validation Loss')
        plt.title('Loss Over Epochs', fontsize=self.title_font)
        plt.xlabel('Epoch', fontsize=self.title_font)
        plt.ylabel('Loss', fontsize=self.title_font)
        plt.tick_params(axis='both', which='major', labelsize=self.title_font)
        plt.legend(fontsize=self.title_font)
        self._save_or_show('loss')

    def _plot_accuracy(self):
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        plt.plot(self._train_dict['accuracy'], label='Train Accuracy')
        plt.plot(self._val_dict['accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs', fontsize=self.title_font)
        plt.xlabel('Epoch', fontsize=self.title_font)
        plt.ylabel('Accuracy', fontsize=self.title_font)
        plt.tick_params(axis='both', which='major', labelsize=self.title_font)
        plt.legend(fontsize=self.title_font)
        self._save_or_show('accuracy')

    def _plot_auc(self):
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        sns.lineplot(data=self._train_dict['auc'], label='Train AUC')
        sns.lineplot(data=self._val_dict['auc'], label='Validation AUC')
        plt.title('AUC Over Epochs', fontsize=self.title_font)
        plt.xlabel('Epoch', fontsize=self.title_font)
        plt.ylabel('AUC', fontsize=self.title_font)
        plt.tick_params(axis='both', which='major', labelsize=self.title_font)
        plt.legend(fontsize=self.title_font)
        self._save_or_show('auc')

    def _plot_mcc(self):
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        sns.lineplot(data=self._train_dict['mcc'], label='Train MCC')
        sns.lineplot(data=self._val_dict['mcc'], label='Validation MCC')
        plt.title('MCC Over Epochs', fontsize=self.title_font)
        plt.xlabel('Epoch', fontsize=self.title_font)
        plt.ylabel('MCC', fontsize=self.title_font)
        plt.tick_params(axis='both', which='major', labelsize=self.title_font)
        plt.legend(fontsize=self.title_font)
        self._save_or_show('mcc')

    def _save_or_show(self, filename):
        """
              Save the currently active matplotlib plot to a PDF file or display it on screen.

              This method decides whether to save the current matplotlib plot based on the presence
              of a directory path in `_fig_path`. If `_fig_path` is set, the plot is saved as a PDF
              file in the specified directory with the given filename. If `_fig_path` is None,
              the plot is displayed using `plt.show()`. After saving or showing the plot, it is closed
              using `plt.close()` to free up memory.

              Parameters
              ----------
              filename : str
                  The name of the file to save the plot as, without the extension. Used only if `_fig_path`
                  is not None.

              Attributes
              ----------
              _save_path : str or None
                  The directory path where the plot should be saved. If None, the plot is displayed on screen.

              Returns
              -------
              None
              """
        if self._save_path:
            full_path = f"{self._save_path}/{filename}.pdf"
            plt.savefig(full_path, format="pdf")
            mlflow.log_artifact(full_path)
        else:
            plt.show()
        plt.close()


def single_datashift_plotter(shift_result, shift_type, save_path):
    """
        Plot performance and uncertainty metrics against dataset shifts.

        This function takes a dictionary of results from a dataset shift experiment, a shift type, and a save path.
        It plots the performance (accuracy and AUC) and uncertainty (Brier score, ECE, and NLL) metrics against the dataset shifts.
        The plots are saved in the specified save path.

        Parameters
        ----------
        shift_result : dict
            The results from the dataset shift experiment. The keys are the dataset shifts, and the values are dictionaries
            with 'performance' and 'uncertainty' keys. The 'performance' dictionary should have 'accuracy' and 'auc' keys,
            and the 'uncertainty' dictionary should have 'brier', 'ece', and 'nll' keys.
        shift_type : str
            The type of dataset shift.
        save_path : str
            The path where the plots should be saved.

        Returns
        -------
        None
    """
    dataset_shifts = [shift for shift in shift_result]
    accuracy = [shift_result[shift]['performance']['accuracy'] for shift in dataset_shifts]
    auc = [shift_result[shift]['performance']['auc'] for shift in dataset_shifts]
    brier = [shift_result[shift]['uncertainty']['brier'] for shift in dataset_shifts]
    ece = [shift_result[shift]['uncertainty']['ece'] for shift in dataset_shifts]
    nll = [shift_result[shift]['uncertainty']['nll'] for shift in dataset_shifts]

    # Setting the style
    sns.set(style="darkgrid")

    save_path = check_folder(path=save_path, path_ext="figures")
    save_path = f"{save_path}/{shift_type}"

    create_plot(x=dataset_shifts, y=auc, y_label='AUC', title=f'{shift_type}: Dataset Shift vs. AUC',
                file_name=f'{save_path}_auc.eps', color="blue")
    create_plot(x=dataset_shifts, y=accuracy, y_label='Accuracy', title=f'{shift_type}: Dataset Shift vs. Accuracy',
                file_name=f'{save_path}_accuracy.eps', color="green")
    create_plot(x=dataset_shifts, y=brier, y_label='Brier Score', title=f'{shift_type}: Dataset Shift vs. Brier Score',
                file_name=f'{save_path}_brier.eps', color='red')
    create_plot(x=dataset_shifts, y=ece, y_label='ECE', title=f'{shift_type}: Dataset Shift vs. ECE',
                file_name=f'{save_path}_ece.eps', color='purple')
    create_plot(x=dataset_shifts, y=nll, y_label='NLL', title=f'{shift_type}: Dataset Shift vs. NLL',
                file_name=f'{save_path}_nll.eps', color='orange')


def create_plot(x, y, y_label, title, file_name, color="darkgreen"):
    plt.figure(figsize=FIG_SIZE)
    sns.lineplot(x=x, y=y, marker='o', linestyle='-', color=color)
    plt.title(f'{title}', fontsize=TITLE_FONT, weight='bold', color='navy')
    plt.xlabel('Dataset Shift', fontsize=LABEL_FONT)
    plt.ylabel(y_label, fontsize=LABEL_FONT)
    plt.xticks(x, fontsize=TICK_FONT)
    plt.yticks(fontsize=TICK_FONT)
    plt.legend(fontsize=TICK_FONT)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, format="eps")


def multiple_datashift_plotter(shift_result, save_path):
    plots = ["accuracy", "auc", "brier", "ece", "nll"]
    sns.set(style="darkgrid")
    save_path = check_folder(path=save_path, path_ext="figures")
    save_path = f"{save_path}/multiple_shifts"

    for p in plots:
        plt.figure(figsize=FIG_SIZE)
        for shift_type in shift_result:
            if p in ("accuracy", "auc"):
                metric = "performance"
            else:
                metric = "uncertainty"
            dataset_shifts = [shift for shift in shift_result[shift_type]]
            y = [shift_result[shift_type][shift][metric][p] for shift in dataset_shifts]
            sns.lineplot(x=dataset_shifts, y=y, marker='o', linestyle='-', label=shift_type)
        sns.despine()
        plt.title(f'{p.upper()}: Over Dataset Shifts', fontsize=TITLE_FONT, weight='bold')
        plt.xlabel('Dataset Shift', fontsize=LABEL_FONT)
        plt.ylabel(p.upper(), fontsize=LABEL_FONT)
        plt.xticks(dataset_shifts, fontsize=TICK_FONT)
        plt.yticks(fontsize=TICK_FONT)
        plt.legend(fontsize=TICK_FONT)
        plt.tight_layout()
        plt.savefig(f"{save_path}_{p}.eps", format="eps", dpi=300)


def multiple_runs_plotter(ensemble_names, ensemble_results, save_path):
    """
    Plot the results of multiple ensemble runs. The results are plotted against the dataset shifts.
    The plots are saved in the specified save path.
    Parameters
    ----------
    ensemble_names: list
        names of the ensembles to be plotted
    ensemble_results: list
        list of dictionaries containing the results of the ensemble runs
    save_path: str
        path where the plots should be saved

    Returns
    -------
    None
    """
    datashifts = list(ensemble_results[0].keys())

    plots = ["accuracy", "auc", "brier", "ece", "nll"]

    for d_shift in datashifts:
        for p in plots:
            if p in ("accuracy", "auc"):
                metric = "performance"
            else:
                metric = "uncertainty"

            # Plotting the results
            plt.figure(figsize=FIG_SIZE)
            for i, name in enumerate(ensemble_names):
                current_datashift = ensemble_results[i][d_shift]
                x = list(current_datashift.keys())
                y = [current_datashift[amount_shift][metric][p] for amount_shift in x]
                print(f"y: {y}")
                sns.lineplot(x=x, y=y, marker='o', linestyle='-', label=name)

            sns.despine()
            plt.title(f'{d_shift.upper()}: {p.upper()}', fontsize=TITLE_FONT, weight='bold')
            plt.xlabel('Dataset Shift', fontsize=LABEL_FONT)
            plt.ylabel(p.upper(), fontsize=LABEL_FONT)
            plt.xticks(x, fontsize=TICK_FONT)
            plt.yticks(fontsize=TICK_FONT)
            plt.legend(fontsize=TICK_FONT)
            plt.tight_layout()
            plt.savefig(f"{save_path}_{d_shift}_{p}.eps", format="eps", dpi=300)
