from typing import Dict, List, Union, Mapping, Any
from acme.utils.loggers import Logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import pickle

LoggingData = Mapping[str, Any]


class ListLogger(Logger):
    """Manually save the data to the class in a dict. Currently only supports scalar history
    inputs."""
    def __init__(self, save: bool = True, save_path: str = "/tmp/logging_hist.pkl",
                 save_period: int = 100):

        self.save = save
        self.save_path = save_path
        if save:
            if not pathlib.Path(self.save_path).parent.exists():
                pathlib.Path(self.save_path).parent.mkdir(exist_ok=True, parents=True)
        self.save_period = save_period  # how often to save the logging history
        self.history: Dict[str, List[Union[np.ndarray, float, int]]] = {}
        self.print_warning: bool = False
        self.iter = 0

    def write(self, data: LoggingData) -> None:
        for key, value in data.items():
            if key in self.history:
                try:
                    value = float(value)
                except:
                    pass
                self.history[key].append(value)
            else:  # add key to history for the first time
                if isinstance(value, np.ndarray):
                    assert np.size(value) == 1
                    value = float(value)
                else:
                    if isinstance(value, float) or isinstance(value, int):
                        pass
                    else:
                        if not self.print_warning:
                            print("non numeric history values being saved")
                            self.print_warning = True
                self.history[key] = [value]

        self.iter += 1
        if self.save and (self.iter + 1) % self.save_period == 0:
            pickle.dump(self.history, open(self.save_path, "wb")) # overwrite with latest version
            print(f"saved latest logging results to {self.save_path}")

    def close(self) -> None:
        if self.save:
            pickle.dump(self.history, open(self.save_path, "wb"))


def plot_history(history):
    """Agnostic history plotter for quickly plotting a dictionary of logging info."""
    figure, axs = plt.subplots(len(history), 1, figsize=(7, 3*len(history.keys())))
    if len(history.keys()) == 1:
        axs = [axs]  # make iterable
    elif len(history.keys()) == 0:
        return
    for i, key in enumerate(history):
        if type(history[key][0]) in [int, float, np.ndarray]:
            if isinstance(history[key][0], np.ndarray):
                if history[key][0].shape not in [(),(1,)]:
                    continue
            data = pd.Series(history[key])

            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            if sum(data.isna()) > 0:
                data = data.dropna()
                print(f"NaN encountered in {key} history")
            axs[i].plot(data)
            axs[i].set_title(key)
    plt.tight_layout()