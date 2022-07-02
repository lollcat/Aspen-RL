import os
import pickle
from datetime import datetime, date
from hydrocarbon_problem.agents.logger import plot_history
import matplotlib.pyplot as plt

if __name__ == '__main__':
    agent = "random"

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    today = date.today()
    print(os.getcwd())

    if agent == "sac":
        os.chdir("../results/SAC")
        path_to_saved_hist = "C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/AspenSimulation/results/logging_hist_sac.pkl"  # path to where history was saved
    elif agent == "random":
        os.chdir("../results/Random")
        path_to_saved_hist = "C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/AspenSimulation/results/logging_hist_random_agent.pkl"
        # "../results/logging_hist_random_agent.pkl"
    hist = pickle.load(open(path_to_saved_hist, "rb"))
    plot_history(hist)
    plt.savefig(f'logger_{agent}_episodes-2000_{today}_{current_time}.pdf')
    plt.show()

    # print(os.getcwd())
    # os.chdir("../results")
    # print(os.getcwd())
    # path_to_saved_hist = "logging_hist.pkl" # path to where history was saved
    # hist = pickle.load(open(path_to_saved_hist, "rb"))
    # plot_history(hist)
    # plt.show()
