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
    name = "2022-07-21-17-10-07_logging_hist_SAC_PID_3000_batch_and_NN_64_LR_1e-4"
    if agent == "sac":
        os.chdir("../results/SAC")
        # os.chdir("../agents/sac")
        # path_to_saved_hist = "C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/agents/sac/Fake_API.pkl"
        path_to_saved_hist = f"C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/AspenSimulation" \
                             f"/results/{name}.pkl"  # path to where history was saved
    elif agent == "random":
        os.chdir("../results/Random")
        path_to_saved_hist = f"C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/AspenSimulation/" \
                             f"results/{name}.pkl"

        # "../results/logging_hist_random_agent.pkl"
    hist = pickle.load(open(path_to_saved_hist, "rb"))

    hist_keys = list(hist.keys())
    agent_spec = {agent_par: hist[agent_par] for agent_par in hist_keys[4:-1]}
    # a=agent_spec["Contact"]
    # a[0] = 1.0
    # agent_spec["Contact"] = a
    # agent_spec.pop("Unconverged")
    col_spec = {col_par: hist[col_par] for col_par in hist.keys() & hist_keys[:4]}

    plot_history(agent_spec)
    # plot_history(hist)
    plt.savefig(f'{name}.pdf')
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





    tops1 = tops[0]
    tops11 = tops1.specification.molar_flows
    tops2 = tops[1]
    tops21 = tops2.specification.molar_flows
    tops3 = tops[2]
    tops31 = tops3.specification.molar_flows
    tops4 = tops[3]
    tops41 = tops4.specification.molar_flows
    tops5 = tops[4]
    tops51 = tops5.specification.molar_flows
    tops6 = tops[5]
    tops61 = tops6.specification.molar_flows
    tops7 = tops[6]
    tops71 = tops7.specification.molar_flows
    tops8 = tops[7]
    tops81 = tops8.specification.molar_flows

    concT1 = []
    for z in tops11:
        total_flows = sum(tops11)
        conc_ = z / total_flows * 100
        concT1.append(conc_)
    print(f"Top1: {concT1}")

    concT2 = []
    for z in tops21:
        total_flows = sum(tops21)
        conc_ = z / total_flows * 100
        concT2.append(conc_)
    print(f"Top2:{concT2}")
    concT3 = []
    for z in tops31:
        total_flows = sum(tops31)
        conc_ = z / total_flows * 100
        concT3.append(conc_)
    print(f"Top3:{concT3}")
    concT4 = []
    for z in tops41:
        total_flows = sum(tops41)
        conc_ = z / total_flows * 100
        concT4.append(conc_)
    print(f"Top4:{concT4}")
    concT5 = []
    for z in tops51:
        total_flows = sum(tops51)
        conc_ = z / total_flows * 100
        concT5.append(conc_)
    print(f"Top5:{concT5}")
    concT6 = []
    for z in tops61:
        total_flows = sum(tops61)
        conc_ = z / total_flows * 100
        concT6.append(conc_)
    print(f"Top6:{concT6}")
    concT7 = []
    for z in tops71:
        total_flows = sum(tops71)
        conc_ = z / total_flows * 100
        concT7.append(conc_)
    print(f"Top7:{concT7}")
    concT8 = []
    for z in tops81:
        total_flows = sum(tops81)
        conc_ = z / total_flows * 100
        concT8.append(conc_)
    print(f"Top8:{concT8}")

    bots1 = bots[0]
    bots11 = bots1.specification.molar_flows
    bots2 = bots[1]
    bots21 = bots2.specification.molar_flows
    bots3 = bots[2]
    bots31 = bots3.specification.molar_flows
    bots4 = bots[3]
    bots41 = bots4.specification.molar_flows
    bots5 = bots[4]
    bots51 = bots5.specification.molar_flows
    bots6 = bots[5]
    bots61 = bots6.specification.molar_flows
    bots7 = bots[6]
    bots71 = bots7.specification.molar_flows
    bots8 = bots[7]
    bots81 = bots8.specification.molar_flows

    concB1 = []
    for z in bots11:
        total_flows = sum(bots11)
        conc_ = z / total_flows * 100
        concB1.append(conc_)
    print(f"Top1: {concB1}")

    concB2 = []
    for z in bots21:
        total_flows = sum(bots21)
        conc_ = z / total_flows * 100
        concB2.append(conc_)
    print(f"Top2:{concB2}")
    concB3 = []
    for z in bots31:
        total_flows = sum(bots31)
        conc_ = z / total_flows * 100
        concB3.append(conc_)
    print(f"Top3:{concB3}")
    concB4 = []
    for z in bots41:
        total_flows = sum(bots41)
        conc_ = z / total_flows * 100
        concB4.append(conc_)
    print(f"Top4:{concB4}")
    concB5 = []
    for z in bots51:
        total_flows = sum(bots51)
        conc_ = z / total_flows * 100
        concB5.append(conc_)
    print(f"Top5:{concB5}")
    concB6 = []
    for z in bots61:
        total_flows = sum(bots61)
        conc_ = z / total_flows * 100
        concB6.append(conc_)
    print(f"Top6:{concB6}")
    concB7 = []
    for z in bots71:
        total_flows = sum(bots71)
        conc_ = z / total_flows * 100
        concB7.append(conc_)
    print(f"Top7:{concB7}")
    concB8 = []
    for z in bots81:
        total_flows = sum(bots81)
        conc_ = z / total_flows * 100
        concB8.append(conc_)
    print(f"Top8:{concB8}")






    for q in tops:
        a = q
        b = a[0]
        flows = b[2]
        total_flows = sum(flows)
        conc = []
        for z in flows:
            conc_top = z / total_flows * 100
            conc.append(conc_top)
            # print(conc_top)






    print("done")
    a = tops[0]
    b = a[0]
    c = b[2]
    print(c)
    concentration_tops = []
    concentrattion_bots = []

    for q in tops:
        a = q
        b = a[0]
        flows = b[2]
        total_flows = sum(flows)
        for z in flows:
            conc_top = z / total_flows * 100
            concentration_tops.append(conc_top)
            print(conc_top)


    for q in bots:
        for w in q:
            for t in w:
                total_flow = sum(t)
                for s in t:
                    conc = s/total_flow * 100
                    concentration_bots.append(conc)

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
