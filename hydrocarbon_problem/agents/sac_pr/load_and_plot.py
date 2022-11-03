import pickle
from hydrocarbon_problem.agents.logger import plot_history
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path_to_saved_hist = "results/logging_hist.pkl" # path to where history was saved
    hist = pickle.load(open(path_to_saved_hist, "rb"))
    plot_history(hist)
    plt.show()
