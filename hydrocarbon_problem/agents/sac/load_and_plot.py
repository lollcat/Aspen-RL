import os
import pickle
from datetime import datetime, date
from hydrocarbon_problem.agents.logger import plot_history
import matplotlib.pyplot as plt

if __name__ == '__main__':
    agent = "sac"

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    today = date.today()
    print(os.getcwd())

    if agent == "sac":
        os.chdir("../results/SAC")
        path_to_saved_hist = "C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/AspenSimulation" \
                             "/results/2022-07-14-20-32-06_logging_hist_DDPG_3000_scaled_reward_batch_and_NN_64.pkl"  # path to where history was saved
    elif agent == "random":
        os.chdir("../results/Random")
        path_to_saved_hist = "C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/AspenSimulation/results/logging_hist_random_agent.pkl"
        # "../results/logging_hist_random_agent.pkl"
    hist = pickle.load(open(path_to_saved_hist, "rb"))

    hist_keys = list(hist.keys())
    agent_spec = {agent_par: hist[agent_par] for agent_par in hist_keys[3:]}
    col_spec = {col_par: hist[col_par] for col_par in hist.keys() & hist_keys[:3]}

    plot_history(agent_spec)
    plt.savefig(f'blob.pdf')
    plt.show()

    """Index for episode with heighest profit"""
    episodic_returns = agent_spec.get("episode_return")
    max_return = max(episodic_returns)
    index_max_return = episodic_returns.index(max_return)  # Episode with the highest return

    

    # plot_history(hist)
    # plt.savefig(f'2022-07-14-20-32-06_logging_hist_DDPG_3000_scaled_reward_batch_and_NN_64.pdf')
    # plt.show()

    # print(os.getcwd())
    # os.chdir("../results")
    # print(os.getcwd())
    # path_to_saved_hist = "logging_hist.pkl" # path to where history was saved
    # hist = pickle.load(open(path_to_saved_hist, "rb"))
    # plot_history(hist)
    # plt.show()
