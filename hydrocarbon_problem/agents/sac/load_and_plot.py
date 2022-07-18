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
                             "/results/2022-07-18-15-38-37_logging_hist_SAC_PID_3000_batch_and_NN_64_LR_1e-3.pkl"  # path to where history was saved
    elif agent == "random":
        os.chdir("../results/Random")
        path_to_saved_hist = "C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/AspenSimulation/" \
                             "results/2022-07-18_11-45-52_logging_hist_random_agent_3000_scaled_reward.pkl"
        # "../results/logging_hist_random_agent.pkl"
    hist = pickle.load(open(path_to_saved_hist, "rb"))

    hist_keys = list(hist.keys())
    agent_spec = {agent_par: hist[agent_par] for agent_par in hist_keys[4:-1]}
    a=agent_spec["Contact"]
    a[0] = 1.0
    agent_spec["Contact"] = a
    agent_spec.pop("Unconverged")
    col_spec = {col_par: hist[col_par] for col_par in hist.keys() & hist_keys[:4]}

    plot_history(agent_spec)
    plt.savefig(f'2022-07-18-15-38-37_logging_hist_SAC_PID_3000_batch_and_NN_64_LR_1e-3.pdf')
    plt.show()

    """Index for episode with heighest profit"""
    episodic_returns = agent_spec.get("episode_return")
    max_return = max(episodic_returns)
    index_max_return = episodic_returns.index(max_return)  # Episode with the highest return

    """Retrieve column/stream specs at max return"""
    tops = []
    bots = []
    col = []
    for i in col_spec:
        for k in col_spec[i]:
            if k.episode == index_max_return:
                if "Bot" in i:
                    bots.append(k)
                elif "Col" in i:
                    col.append(k)
                elif "Top" in i:
                    tops.append(k)


    # col_max_return = []
    # for i in col_spec:
    #     for k in col_spec[i]:
    #         if k.episode == index_max_return:
    #             col_max_return.append([k])



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
