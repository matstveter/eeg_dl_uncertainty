import os
import pickle
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, Listbox, END, Button, Toplevel

from eegDlUncertainty.data.results.plotter import multiple_datashift_plotter, multiple_runs_plotter, \
    single_datashift_plotter


class DirectorySelectDialog(Toplevel):
    def __init__(self, parent, initial_dir=None):
        Toplevel.__init__(self, parent)
        self.title("Select Directories")
        self.geometry("400x300")
        self.resizable(False, False)

        self.listbox = Listbox(self)
        self.listbox.pack(pady=15, padx=15, fill='both', expand=True)

        self.add_button = Button(self, text="Add Directory", command=self.add_directory)
        self.add_button.pack(side='left', padx=(20, 0))

        self.ok_button = Button(self, text="OK", command=self.ok)
        self.ok_button.pack(side='right', padx=(0, 20))
        self._initial_dir = initial_dir

        self.selected_directories = []

    def add_directory(self):
        directory = filedialog.askdirectory(initialdir=self._initial_dir)
        if directory:
            self.listbox.insert(END, directory)
            self.selected_directories.append(directory)

    def ok(self):
        self.destroy()


def get_folder(result_path):
    root = tk.Tk()
    root.withdraw()
    dialog = DirectorySelectDialog(root, initial_dir=result_path)
    root.wait_window(dialog)
    selected_directories = dialog.selected_directories

    root.destroy()
    return selected_directories


def get_run_data(path):
    with open(path + "/datashift_results.pkl", "rb") as f:
        data = dict(pickle.load(f))

    new_dict = {}
    # Only interested in the results and not the predictions
    for key, val in data.items():
        new_dict[key] = val['results']

    return new_dict


def get_plots_one_run(shift_results, save_path):
    single_plot_path = os.path.join(save_path, "single_plots")
    os.makedirs(single_plot_path, exist_ok=True)
    merged_plot_path = os.path.join(save_path, "merged_plots")
    os.makedirs(merged_plot_path, exist_ok=True)

    for shift_type, shift_result in shift_results.items():
        single_datashift_plotter(shift_result=shift_result, shift_type=shift_type, save_path=single_plot_path)

    multiple_datashift_plotter(shift_result=shift_results, save_path=merged_plot_path)



if __name__ == '__main__':
    result_path = "/home/tvetern/PhD/dl_uncertainty/results/"
    
    folder_name = f"generated_plots_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}/"
    save_path = os.path.join("./plots/", folder_name)
    os.makedirs(save_path, exist_ok=True)

    folders_to_plot = ["/home/tvetern/PhD/dl_uncertainty/results/weight_ensemble_2024-06-27 14_26_50",
                       "/home/tvetern/PhD/dl_uncertainty/results/bagging_ensemble_2024-06-27 12_58_30"]

    if len(folders_to_plot) == 0:
        folders_to_plot = get_folder(result_path)

    if len(folders_to_plot) == 1:
        shift_results = get_run_data(folders_to_plot[0])
        get_plots_one_run(shift_results, save_path=save_path)
    else:
        names = []
        results = []
        for f in folders_to_plot:
            ensemble_name = os.path.basename(f).split("_")[0]
            i = 0
            while ensemble_name in names:
                ensemble_name = ensemble_name + "_" + str(i)
                i += 1

            names.append(ensemble_name)
            results.append(get_run_data(f))

        multiple_runs_plotter(ensemble_names=names, ensemble_results=results, save_path=save_path)
