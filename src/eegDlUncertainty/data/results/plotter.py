import os.path
import warnings

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns


class Plotter:

    def __init__(self, train_dict, val_dict, test_dict, test_dict_best_model, save_path=None):
        self._train_dict = train_dict
        self._val_dict = val_dict
        self._test_dict = test_dict
        self._test_dict_best_model = test_dict_best_model

        if save_path is not None:
            self._save_path = self._check_folder(path=save_path)
        else:
            self._save_path = save_path

    @staticmethod
    def _check_folder(path):
        path_ext = "figures"

        full_path = os.path.join(path, path_ext)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)

        return full_path

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
        plt.figure(figsize=(10, 6))
        plt.plot(self._train_dict['loss'], label='Train Loss')
        plt.plot(self._val_dict['loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        self._save_or_show('loss')

    def _plot_accuracy(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self._train_dict['accuracy'], label='Train Accuracy')
        plt.plot(self._val_dict['accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        self._save_or_show('accuracy')

    def _plot_auc(self):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self._train_dict['auc'], label='Train AUC')
        sns.lineplot(data=self._val_dict['auc'], label='Validation AUC')
        plt.title('AUC Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        self._save_or_show('auc')

    def _plot_mcc(self):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self._train_dict['mcc'], label='Train MCC')
        sns.lineplot(data=self._val_dict['mcc'], label='Validation MCC')
        plt.title('MCC Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MCC')
        plt.legend()
        self._save_or_show('mcc')

    def _save_or_show(self, filename):
        if self._save_path:
            full_path = f"{self._save_path}/{filename}.pdf"
            plt.savefig(full_path, format="pdf")
            mlflow.log_artifact(full_path)
        else:
            plt.show()
        plt.close()
