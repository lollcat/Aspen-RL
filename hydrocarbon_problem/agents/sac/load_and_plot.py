import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date
from hydrocarbon_problem.agents.logger import plot_history
import matplotlib.pyplot as plt


if __name__ == '__main__':
    agent = "sac"
    load_two_hists = False
    create_png = True
    column_table = False
    stream_table = False
    moving_average = True
    reward_scale = 10
    version = "new"
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    today = date.today()
    print(os.getcwd())

    path = r"../results/2022-09-16-11-13-04\SAC_ActionSpace_Full_logger_20000_LRalpha7_5e-5_SAC_updates_4_steps_4"
    save_location = f"../{path[3:30]}"  # 2022-09-02-12-01-31"
    name = path[31:]  # "SAC_ActionSpace_Full_basecase_bounds_logger_20000_LR3e-4_SAC_updates_1"
    if load_two_hists:
        path2 = r"../results/2022-09-05-10-35-28\SAC_ActionSpace_Full_basecase_bounds_logger_20000_LR3e-4_SAC_updates_4_continue_run02-09-2022_12_04_34"
        save_location2 = f"../{path2[3:30]}"
        name2 = path[31:]
    else:
        pass

    if agent == "sac":
        # os.chdir(save_location)
        # os.chdir("../results/2022-08-10-10-03-33")
        # path_to_saved_hist = "C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/agents/sac/Fake_API.pkl"
        path_to_saved_hist = (f'{os.path.abspath(path)}.pkl')
        if load_two_hists:
            path_to_saved_hist2 = (f'{os.path.abspath(path2)}.pkl')
        # path_to_saved_hist =(f"C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/AspenSimulation" /
        #                      f"/results/{name}.pkl")  # path to where history was saved
    elif agent == "random":
        os.chdir(save_location)
        # path_to_saved_hist = f"C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/AspenSimulation/" \
        #                      f"results/{name}.pkl"
        path_to_saved_hist = f"{path}.pkl"
        # "../results/logging_hist_random_agent.pkl"

    hist = pickle.load(open(path_to_saved_hist, "rb"))
    print(f"Mean: {np.mean(hist['episode_return'])*reward_scale}")
    print(f"Max: {max(hist['episode_return'])*reward_scale}")
    print(f"Number of episodes: {len(hist['episode_return'])}")
    if not load_two_hists:
        os.chdir(save_location)
    if load_two_hists:
        hist2 = pickle.load(open(path_to_saved_hist2, "rb"))
        print(f"Mean2: {np.mean(hist2['episode_return']) * reward_scale}")
        print(f"Max2: {max(hist2['episode_return']) * reward_scale}")
        print(f"Number of episodes2: {len(hist2['episode_return'])}")
        os.chdir(save_location2)
        merged_returns = hist['episode_return'] + hist2['episode_return']
        print(f"Mean_merged: {np.mean(merged_returns) * reward_scale}")
        print(f"Max_merged: {max(merged_returns) * reward_scale}")
        print(f"Number of episodes_merged: {len(merged_returns)}")




    episodic_separate1 = hist.get("Separate")
    non_separate_index1 = []

    for i in range(len(episodic_separate1)):
        if episodic_separate1[i] == 0:
            non_separate_index1.append(i)
    non_separate_convergence_values1 = []
    episodic_convergence1 = hist.get("Converged")
    for i in non_separate_index1:
        non_separate_convergence_values1.append(episodic_convergence1[i])

    z1 = non_separate_convergence_values1.count(2)  # number_of_non_separations_with_convergence_of_2
    x1 = non_separate_convergence_values1.count(1)  # number_of_non_separations_with_convergence_of_1
    y1 = non_separate_convergence_values1.count(0)  # number_of_non_separations_with_convergence_of_0
    w1 = non_separate_convergence_values1.count(-1)  # number_of_non_separations_with_convergence_of_-1
    print(z1)
    print(x1)
    print(y1)
    print(w1)

    a1 = hist["Separate"].count(1)
    b1 = hist["Separate"].count(0)
    c1 = hist['Converged'].count(0) - y1
    d1 = hist['Converged'].count(1) - x1
    e1 = hist['Converged'].count(2) - z1
    f1 = hist['Converged'].count(-1) - w1
    print(f"Separation1 Y: {a1}")
    print(f"Separation1 N: {b1}")
    print(f"Converged1: {c1}")
    print(f"Errors1: {d1}")
    print(f"Converged with warnings1: {e1}")
    print(f"Lost contact1: {f1}")
    hist_keys = list(hist.keys())

    if load_two_hists:



        episodic_separate2 = hist2.get("Separate")
        non_separate_index2 = []

        for i in range(len(episodic_separate2)):
            if episodic_separate2[i] == 0:
                non_separate_index2.append(i)
        non_separate_convergence_values2 = []
        episodic_convergence2 = hist2.get("Converged")
        for i in non_separate_index2:
            non_separate_convergence_values2.append(episodic_convergence2[i])

        z2 = non_separate_convergence_values2.count(2)  # number_of_non_separations_with_convergence_of_2
        x2 = non_separate_convergence_values2.count(1)  # number_of_non_separations_with_convergence_of_1
        y2 = non_separate_convergence_values2.count(0)  # number_of_non_separations_with_convergence_of_0
        w2 = non_separate_convergence_values2.count(-1)  # number_of_non_separations_with_convergence_of_-1
        print(z2)
        print(x2)
        print(y2)
        print(w2)

        a2 = hist2["Separate"].count(1)
        b2 = hist2["Separate"].count(0)
        c2 = hist2['Converged'].count(0) - y2
        d2 = hist2['Converged'].count(1) - x2
        e2 = hist2['Converged'].count(2) - z2
        f2 = hist2['Converged'].count(-1) - w2
        print(f"Separation2 Y: {a2}")
        print(f"Separation2 N: {b2}")
        print(f"Converged2: {c2}")
        print(f"Errors2: {d2}")
        print(f"Converged with warnings2: {e2}")
        print(f"Lost contact2: {f2}")

        print(f"Separation_merged Y: {a1+a2}")
        print(f"Separation_merged N: {b1+b2}")
        print(f"Converged_merged: {c1+c2}")
        print(f"Errors_merged: {d1+d2}")
        print(f"Converged with warnings_merged: {e1+e2}")
        print(f"Lost contact_merged: {f1+f2}")

    if moving_average and not load_two_hists:
        number_of_episodes_hist = len(hist['episode_return'])
        episodes = list(range(0, number_of_episodes_hist))
        return_ = hist['episode_return']
        return_ = [i * reward_scale for i in return_]
        data = {'Episodes': episodes,
                'Return': return_}
        df_return = pd.DataFrame(data)
        # MA = df_return.rolling(window=2000, center=False).mean()
        df_return['Moving average'] = df_return['Return'].rolling(window=100, center=True).mean()
        # df_return['Moving average'] = df_return.rolling(window=10, center=True).mean()
        df_return.plot.line(x='Episodes', y=['Return', 'Moving average'])
        plt.title(f'Episodic return and moving average')
        plt.ylabel('Return [M€]')
        plt.xlabel('Episode')
        plt.savefig(f'Return_{name}.png')
        plt.show()
        plt.close()
    elif moving_average and load_two_hists:
        number_of_episodes_hist = len(hist['episode_return'])
        number_of_episodes_hist2 = len(hist2['episode_return'])
        last_checkpoint_episode = number_of_episodes_hist - (number_of_episodes_hist % 500)
        episodes = list(range(0, last_checkpoint_episode + number_of_episodes_hist2))

        return_ = hist['episode_return']
        return_ = return_[:last_checkpoint_episode]
        return_ = [i * reward_scale for i in return_]
        return2_ = hist2['episode_return']
        return2_ = [i * reward_scale for i in return2_]
        merged_return = return_ + return2_
        print(f"Mean_merged: {np.mean(merged_return)* reward_scale}")
        print(f"Max_merged: {max(merged_return)* reward_scale}")
        print(f"Number of episodes_merged: {len(episodes)}")

        data = {'Episodes': episodes,
                'Return': merged_return}
        df_return = pd.DataFrame(data)
        # MA = df_return.rolling(window=2000, center=False).mean()
        df_return['Moving average'] = df_return['Return'].rolling(window=100, center=True).mean()
        # df_return['Moving average'] = df_return.rolling(window=10, center=True).mean()
        df_return.plot.line(x='Episodes', y=['Return', 'Moving average'])
        plt.title(f'Episodic return and moving average')
        plt.ylabel('Return [M€]')
        plt.xlabel('Episode')
        plt.savefig(f'merged_Return_{name}.png')
        plt.show()
        plt.close()

    if agent == 'sac':
        agent_keys = {'episode_return', 'Revenue' ,'episode_time', 'actor_loss', 'actor_loss__log_prob_mean',
                      'actor_loss__min_q_mean', 'alpha', 'alpha_loss',
                      'critic_loss', 'critic_loss__next_log_prob_mean', 'critic_loss__next_q_mean',
                      'critic_loss__target_q_mean', 'observations_mean',  'observation_std', 'agent_step_time', 'Separate',
                      'Streams yet to be acted on', 'Converged'}
        if version == "old":
            column_keys = {'Diameter', 'Height', 'n_stages', 'feed_stage_location', 'reflux_ratio', 'reboil_ratio', 'condenser_pressure', }

        elif version == "new":
            column_keys = {'Diameter', 'Height', 'n_stages', 'feed_stage_location', 'reflux_ratio', 'reboil_ratio',
                           'condenser_pressure', "a_cnd", "a_rbl", "cost_col", "cost_int", "cost_cnd", "cost_rbl",
                           "cost_util_cnd", "cost_util_rbl", "RR FirstStream", "RB FirstStream"}
        time_keys = {'Time to set aspen', 'Time to run aspen', 'Time to retrieve aspen data', 'Time to calculate reward',
                     'time_to_sample_from_buffer', 'time_to_update_agent'}

        agent_dict = {key: value for key, value in hist.items() if key in agent_keys}
        if load_two_hists:
            agent_dict2 = {key: value for key, value in hist2.items() if key in agent_keys}
            agent_dict2['episode_return'][0] = agent_dict2['episode_return'][0].item()
            column_dict2 = hist2["Column"]
            time_dict2 = {key: value for key, value in hist2.items() if key in time_keys}
            if agent_dict2['Separate'][0]:
                agent_dict2['Separate'][0] = 1
            else:
                agent_dict2['Separate'][0] = 0
        if version == "old":
            column_dict = {key: value for key, value in hist.items() if key in column_keys}
            column_dict['reflux_ratio'][0] = column_dict['reflux_ratio'][0].item()
            column_dict['reboil_ratio'][0] = column_dict['reboil_ratio'][0].item()
            column_dict['Diameter'][0] = column_dict['Diameter'][0].item()
        elif version == "new":
            # agent_dict['Revenue'][0] = agent_dict['Revenue'][0].item()
            agent_dict['episode_return'][0] = agent_dict['episode_return'][0].item()
            column_dict = hist["Column"]
            # agent_dict2['episode_return'][0] = agent_dict2['episode_return'][0].item()
            # column_dict2 = hist2["Column"]
        time_dict = {key: value for key, value in hist.items() if key in time_keys}


        if agent_dict['Separate'][0]:
            agent_dict['Separate'][0] = 1
        else:
            agent_dict['Separate'][0] = 0

        # plot_history(agent_spec)
        agent_plots = ["actor_loss", "actor_log_prob", "critic_loss", "alpha_loss"]
        if load_two_hists:
            agent_dict_png = {"actor_loss": agent_dict["actor_loss"] + agent_dict2["actor_loss"],
                           "critic_loss": agent_dict["critic_loss"] + agent_dict2["critic_loss"],
                           "alpha_loss": agent_dict["alpha_loss"] + agent_dict2["alpha_loss"]}
        else:
            agent_dict_png = {"actor_loss": agent_dict["actor_loss"],
                              "actor_log_prob": agent_dict["actor_loss__log_prob_mean"],
                              "critic_loss": agent_dict["critic_loss"],
                              "alpha_loss": agent_dict["alpha_loss"]}
        for i in agent_plots:
            if not load_two_hists:
                plt.plot(agent_dict_png[i])
                if i == "actor_loss":
                    naming = "Actor loss"
                elif i== "actor_log_prob":
                    naming = "Actor log prob"
                elif i == "critic_loss":
                    naming = "Critic loss"
                elif i == "alpha_loss":
                    naming = "Alpha loss"
                plt.ylabel(naming)
                plt.title(naming)
                plt.xlabel("Agent update")
                plt.savefig(f"{i}_.png")
                plt.show()
                plt.close()

        plot_history(agent_dict)
        # os.chdir(save_location)
        plt.savefig(f'Agent_data_{name}.pdf')
        plt.show()
        plt.close()

    elif agent == "random":
        column_keys = {'episode_return', 'Revenue' ,'episode_time', 'Diameter', 'Height', 'n_stages', 'feed_stage_location', 'reflux_ratio', 'reboil_ratio',
                       'condenser_pressure', "a_cnd", "a_rbl", "cost_col", "cost_int", "cost_cnd", "cost_rbl",
                       "cost_util_cnd", "cost_util_rbl", "RR FirstStream", "RB FirstStream", 'Separate',
                      'Streams yet to be acted on', 'Converged'}
        time_keys = {'Time to set aspen', 'Time to run aspen', 'Time to retrieve aspen data',
                     'Time to calculate reward',
                     'time_to_sample_from_buffer', 'time_to_update_agent'}


        if version == "old":
            column_dict = {key: value for key, value in hist.items() if key in column_keys}
            column_dict['reflux_ratio'][0] = column_dict['reflux_ratio'][0].item()
            column_dict['reboil_ratio'][0] = column_dict['reboil_ratio'][0].item()
            column_dict['Diameter'][0] = column_dict['Diameter'][0].item()
        elif version == "new":
            column_dict = hist["Column"]
        time_dict = {key: value for key, value in hist.items() if key in time_keys}

    if version == "new":
        n_stages = []
        feed_stage_location = []
        reflux_ratio = []
        reboil_ratio = []
        condenser_pressure = []
        condenser_temperature = []
        reboiler_temperature = []
        condenser_duty = []
        reboiler_duty = []
        diameter = []
        height = []
        if not load_two_hists:
            first_RR = hist["reflux_ratioFirst stream"]
            first_RR[0] = float(first_RR[0])
            first_BR = hist["reboil_ratioFirst stream"]
            first_BR[0] = float(first_BR[0])
            first_n_stages = hist["n_stagesFirst stream"]
            first_cnd_pressure = hist["condensor_pressureFirst stream"]
            first_cnd_pressure[0] = float(first_cnd_pressure[0])
        if load_two_hists:

            first_RR1 = hist["reflux_ratioFirst stream"]
            first_RR1[0] = float(first_RR1[0])
            first_RR1 = first_RR1[:last_checkpoint_episode]
            first_RR2 = hist2["reflux_ratioFirst stream"]
            first_RR2[0] = float(first_RR2[0])
            merged_rr = first_RR1 + first_RR2
            first_BR1 = hist["reboil_ratioFirst stream"]
            first_BR1[0] = float(first_BR1[0])
            first_BR1 = first_BR1[:last_checkpoint_episode]
            first_BR2 = hist2["reboil_ratioFirst stream"]
            first_BR2[0] = float(first_BR2[0])
            merged_br = first_BR1 + first_BR2
            first_n_stages1 = hist["n_stagesFirst stream"]
            first_n_stages2 = hist2["n_stagesFirst stream"]
            first_n_stages1 = first_n_stages1[:last_checkpoint_episode]
            merged_nstages = first_n_stages1 + first_n_stages2
            first_cnd_pressure1 = hist["condensor_pressureFirst stream"]
            first_cnd_pressure1[0] = float(first_cnd_pressure1[0])
            first_cnd_pressure1 = first_cnd_pressure1[:last_checkpoint_episode]
            first_cnd_pressure2 = hist2["condensor_pressureFirst stream"]
            first_cnd_pressure2[0] = float(first_cnd_pressure2[0])
            merged_cnd_press = first_cnd_pressure1 + first_cnd_pressure2

        for k in column_dict:
            n_stages.append(k.input_spec.n_stages)
            feed_stage_location.append(k.input_spec.feed_stage_location)
            reflux_ratio.append(float(k.input_spec.reflux_ratio))
            reboil_ratio.append(float(k.input_spec.reboil_ratio))
            condenser_pressure.append(float(k.input_spec.condensor_pressure))
            condenser_temperature.append(float(k.output_spec.condenser_temperature))
            reboiler_temperature.append(float(k.output_spec.reboiler_temperature))
            condenser_duty.append(float(k.output_spec.condenser_duty))
            reboiler_duty.append(float(k.output_spec.reboiler_duty))
            diameter.append(float(k.diameter))
            height.append(float(k.height))
        if load_two_hists:
            for j in column_dict2:
                n_stages.append(j.input_spec.n_stages)
                feed_stage_location.append(j.input_spec.feed_stage_location)
                reflux_ratio.append(float(j.input_spec.reflux_ratio))
                reboil_ratio.append(float(j.input_spec.reboil_ratio))
                condenser_pressure.append(float(j.input_spec.condensor_pressure))
                condenser_temperature.append(float(j.output_spec.condenser_temperature))
                reboiler_temperature.append(float(j.output_spec.reboiler_temperature))
                condenser_duty.append(float(j.output_spec.condenser_duty))
                reboiler_duty.append(float(j.output_spec.reboiler_duty))
                diameter.append(float(j.diameter))
                height.append(float(j.height))


        if not load_two_hists:
            column_spec = {"height": height,
                           "n_stages": n_stages,
                           "Diameter": diameter,
                           "RR": reflux_ratio,
                           "RB": reboil_ratio,
                           "n_stages FirstStream": first_n_stages,
                           "cnd_pressure FirstStream": first_cnd_pressure,
                           "RR FirstStream": first_RR,
                           "BR FirstStream": first_BR,
                           "Condenser pressure": condenser_pressure,
                           "Condenser temperature": condenser_temperature,
                           "Reboiler temperature": reboiler_temperature,
                           "Condenser duty": condenser_duty,
                           "Reboiler duty": reboiler_duty,
                           }
        elif load_two_hists:
            column_spec = {"height": height,
                           "n_stages": n_stages,
                           "Diameter": diameter,
                           "RR": reflux_ratio,
                           "RB": reboil_ratio,
                           "n_stages FirstStream": merged_nstages,
                           "cnd_pressure FirstStream": merged_cnd_press,
                           "RR FirstStream": merged_rr,
                           "BR FirstStream": merged_br,
                           "Condenser pressure": condenser_pressure,
                           "Condenser temperature": condenser_temperature,
                           "Reboiler temperature": reboiler_temperature,
                           "Condenser duty": condenser_duty,
                           "Reboiler duty": reboiler_duty,
                           }

    if create_png and not load_two_hists:


        plt.plot(first_cnd_pressure)
        plt.ylabel("Condenser pressure [bar]")
        plt.title("Condenser pressure of first column")
        plt.xlabel("Episode")
        plt.savefig(f"CND_Press_.png")
        plt.show()
        plt.close()
        plt.plot(first_RR)
        plt.ylabel("Reflux ratio [-]")
        plt.title("Reflux ratio of first column")
        plt.xlabel("Episode")
        plt.savefig(f"RR_.png")
        plt.show()
        plt.close()
        plt.plot(first_BR)
        plt.ylabel("Boilup ratio [-]")
        plt.title("Boilup ratio of first column")
        plt.xlabel("Episode")
        plt.savefig(f"BR_.png")
        plt.show()
        plt.close()
        plt.plot(condenser_duty)
        plt.ylabel("Condenser duty [MW]")
        plt.title("Condenser duty over all columns")
        plt.xlabel("Aspen run")
        plt.savefig(f"CND_DUTY_.png")
        plt.show()
        plt.close()

        plt.plot(reboiler_duty)
        plt.ylabel("Reboiler duty [MW]")
        plt.title("Reboiler duty over all columns")
        plt.xlabel("Aspen run")
        plt.savefig(f"RBL_DUTY_.png")
        plt.show()
        plt.close()

    if create_png and load_two_hists:
        plt.plot(merged_nstages)
        plt.ylabel("Number of stages")
        plt.title("Number of stages of first column")
        plt.xlabel("Episode")
        plt.savefig(f"n_Stages_.png")
        plt.show()
        plt.close()

        plt.plot(merged_cnd_press)
        plt.ylabel("Condenser pressure [bar]")
        plt.title("Condenser pressure of first column")
        plt.xlabel("Episode")
        plt.savefig(f"CND_Press_.png")
        plt.show()
        plt.close()

        plt.plot(merged_rr)
        plt.ylabel("Reflux ratio [-]")
        plt.title("Reflux ratio of first column")
        plt.xlabel("Episode")
        plt.savefig(f"RR_.png")
        plt.show()
        plt.close()

        plt.plot(merged_br)
        plt.ylabel("Reboil ratio [-]")
        plt.title("Reboil ratio of first column")
        plt.xlabel("Episode")
        plt.savefig(f"BR_.png")
        plt.show()
        plt.close()

        plt.plot(condenser_duty)
        plt.ylabel("Condenser duty [MW]")
        plt.title("Condenser duty over all columns")
        plt.xlabel("Episode")
        plt.savefig(f"CND_DUTY_.png")
        plt.show()
        plt.close()

        plt.plot(reboiler_duty)
        plt.ylabel("Reboiler duty [MW]")
        plt.title("Reboiler duty over all columns")
        plt.xlabel("Episode")
        plt.savefig(f"RBL_DUTY_.png")
        plt.show()
        plt.close()

    plot_history(column_spec)
    # os.chdir(save_location)
    plt.savefig(f'Column_data_{name}.pdf')
    plt.show()
    plt.close()

    plot_history(time_dict)
    # os.chdir(save_location)
    plt.savefig(f'Time_data_{name}.pdf')
    plt.show()
    plt.close()

    """Index for episode with heighest profit"""
    episodic_returns = agent_dict.get("episode_return")
    max_return = max(episodic_returns)
    index_max_return = episodic_returns.index(max_return)  # Episode with the highest return

    """Retrieve column/stream specs at max return"""
    tops = []
    bots = []
    col = []

    separation_keys = {"TopStream", "BottomStream", "Column"}

    separation_train = {key: value for key, value in hist.items() if key in separation_keys}

    for i in separation_train:
        for k in separation_train[i]:
            if k.episode == index_max_return:
                if "Bot" in i:
                    bots.append(k)
                elif "Col" in i:
                    col.append(k)
                elif "Top" in i:
                    tops.append(k)

    if column_table:
        df_col = pd.DataFrame()

        col_number = []
        col_diameter = []
        col_height = []
        col_RR = []
        col_BR = []
        col_pressure = []
        col_a_cnd = []
        col_a_rbl = []
        col_Q_rbl = []
        col_Q_cnd = []
        col_T_rbl = []
        col_T_cnd = []
        col_cost = []
        int_cost = []
        cnd_cost = []
        rbl_cost = []
        cnd_util_cost = []
        rbl_util_cost = []
        col_feed = []
        col_bots = []
        col_tops = []

        for i in col:

            column_number = i.column_number
            column_diameter = i.diameter
            column_height = i.height
            column_rr = i.input_spec.reflux_ratio
            column_br = i.input_spec.reboil_ratio
            column_pressure = i.input_spec.condensor_pressure

            column_Q_cnd = i.output_spec.condenser_duty
            column_Q_rbl = i.output_spec.reboiler_duty
            column_T_cnd = i.output_spec.condenser_temperature
            column_T_rbl = i.output_spec.reboiler_temperature
            if version == "new":
                cnd_area = i.a_cnd
                rbl_area = i.a_rbl
                column_cost = i.cost_col
                internal_cost = i.cost_int
                cnd_cost_val = i.cost_cnd
                rbl_cost_val = i.cost_rbl
                cnd_util_cost_val = i.cost_util_cnd
                rbl_util_cost_val = i.cost_util_rbl
            else:
                pass
            column_feed = i.input_stream_number
            column_top = i.tops_stream_number
            column_bottom = i.bottoms_stream_number

            col_number.append(column_number)
            col_height.append(column_height)
            col_diameter.append(column_diameter)
            col_RR.append(column_rr)
            col_BR.append(column_br)
            col_pressure.append(column_pressure)
            col_Q_cnd.append(column_Q_cnd)
            col_Q_rbl.append(column_Q_rbl)
            col_T_cnd.append(column_T_cnd)
            col_T_rbl.append(column_T_rbl)
            col_feed.append(column_feed)
            col_tops.append(column_top)
            col_bots.append(column_bottom)

            if version == "new":
                col_a_cnd.append(cnd_area)
                col_a_rbl.append(rbl_area)
                col_cost.append(column_cost)
                int_cost.append(internal_cost)
                cnd_cost.append(cnd_cost_val)
                rbl_cost.append(rbl_cost_val)
                cnd_util_cost.append(cnd_util_cost_val)
                rbl_util_cost.append(rbl_util_cost_val)

        if version == "old":
            df_col = pd.DataFrame({"Column number": col_number,
                                   "Diameter": col_diameter,
                                   "Height": col_height,
                                   "Reflux ratio": col_RR,
                                   "Reboil ratio": col_BR,
                                   "Condenser pressure": col_pressure,
                                   "Condenser duty": col_Q_cnd,
                                   "Reboiler duty": col_Q_rbl,
                                   "Condenser temperature": col_T_cnd,
                                   "Reboiler temperature": col_T_rbl,
                                   "Feed stream": col_feed,
                                   "Top stream": col_tops,
                                   "Bottom stream": col_bots})

        elif version == "new":
            df_col = pd.DataFrame({"Column number": col_number,
                                   "Diameter": col_diameter,
                                   "Height": col_height,
                                   "Reflux ratio": col_RR,
                                   "Reboil ratio": col_BR,
                                   "Condenser pressure": col_pressure,
                                   "Condenser area": col_a_cnd,
                                   "Reboiler area": col_a_rbl,
                                   "Condenser duty": col_Q_cnd,
                                   "Reboiler duty": col_Q_rbl,
                                   "Condenser temperature": col_T_cnd,
                                   "Reboiler temperature": col_T_rbl,
                                   "Column cost": col_cost,
                                   "Internal cost": int_cost,
                                   "Condenser cost": cnd_cost,
                                   "Reboiler cost": rbl_cost,
                                   "Condenser util cost": cnd_util_cost,
                                   "Reboiler util cost": rbl_util_cost,
                                   "Feed stream": col_feed,
                                   "Top stream": col_tops,
                                   "Bottom stream": col_bots})

        df_col = df_col.sort_values(by=["Column number"])
        df_col = df_col.T

        # df_col["Height"] = col_height
        #
        # for i in col:
        #     k = i.input_spec.reflux_ratio
        #     col_RR.append(k)
        # df_col["Reflux ratio"] = col_RR
        #
        #
        # for i in col:
        #     k = i.input_spec.reboil_ratio
        #     col_BR.append(k)
        # df_col["Boilup ratio"] = col_BR
        #
        #
        # for i in col:
        #     k = i.input_spec.condensor_pressure
        #     col_pres.append(k)
        # df_col["Condenser pressure"] = col_pres
        #
        #
        # for i in col:
        #     k = i.output_spec.reboiler_duty
        #     col_Q_reb.append(k)
        # df_col["Reboiler duty"] = col_Q_reb
        #
        #
        # for i in col:
        #     k = i.output_spec.reboiler_temperature
        #     col_T_rbl.append(k)
        # df_col["Reboiler temperature"] = col_T_rbl
        #
        #
        # for i in col:
        #     k = i.output_spec.condenser_duty
        #     col_Q_cnd.append(k)
        # df_col["Condenser duty"] = col_Q_cnd
        #
        #
        # for i in col:
        #     k = i.output_spec.condenser_temperature
        #     col_T_cnd.append(k)
        # df_col["Condenser temperature"] = col_T_cnd
        #
        #         for i in col:
        #     k = i.input_stream_number
        #     col_feed.append(k)
        # df_col["Feed number"] = col_feed
        #
        #         for i in col:
        #     k = i.bottoms_stream_number
        #     col_bots.append(k)
        # df_col["Bottom stream #"] = col_bots
        #
        #         for i in col:
        #     k = i.tops_stream_number
        #     col_tops.append(k)
        # df_col["Top stream #"] = col_tops
        #
        # df_col = df_col.T
        # df_col.to_excel("test.xlsx", sheet_name="ColumnTable")
    if stream_table:

        df_streams = pd.DataFrame()

        stream_number = []
        stream_value = []
        stream_outlet = []
        stream_product = []
        stream_temperature = []
        stream_pressure = []
        C2_flow = []
        C3_flow = []
        iC4_flow = []
        nC4_flow = []
        iC5_flow = []
        nC5_flow = []

        C2_conc = []
        C3_conc = []
        iC4_conc = []
        nC4_conc = []
        iC5_conc = []
        nC5_conc = []

        for i in tops, bots:
            for k in i:
                stream_nmbr = k.number
                stream_val = k.value
                stream_is_outlet = k.is_outlet
                stream_is_product = k.is_product
                stream_temp = k.specification.temperature
                stream_pres = k.specification.pressure
                total_flow = sum(k.specification.molar_flows)
                C2 = k.specification.molar_flows.ethane
                C3 = k.specification.molar_flows.propane
                iC4 = k.specification.molar_flows.isobutane
                nC4 = k.specification.molar_flows.n_butane
                iC5 = k.specification.molar_flows.isopentane
                nC5 = k.specification.molar_flows.n_pentane

                conc_C2 = C2/total_flow
                conc_C3 = C3/total_flow
                conc_iC4 = iC4/total_flow
                conc_nC4 = nC4/total_flow
                conc_iC5 = iC5/total_flow
                conc_nC5 = nC5/total_flow

                stream_number.append(stream_nmbr)
                stream_value.append(stream_val)
                stream_outlet.append(stream_is_outlet)
                stream_product.append(stream_is_product)
                stream_temperature.append(stream_temp)
                stream_pressure.append(stream_pres)
                C2_flow.append(C2)
                C3_flow.append(C3)
                iC4_flow.append(iC4)
                nC4_flow.append(nC4)
                iC5_flow.append(iC5)
                nC5_flow.append(nC5)

                C2_conc.append(conc_C2)
                C3_conc.append(conc_C3)
                iC4_conc.append(conc_iC4)
                nC4_conc.append(conc_nC4)
                iC5_conc.append(conc_iC5)
                nC5_conc.append(conc_nC5)

        molars = ["", ] * len(stream_number)
        df_streams = pd.DataFrame({'Stream number': stream_number,
                                   "Is product": stream_product,
                                   "Is outlet": stream_outlet,
                                   "Value": stream_value,
                                   "Temperature [C]": stream_temperature,
                                   "Pressure [bar]": stream_pressure,
                                   "Molar flows [kmol/s]": molars,
                                   "C2": C2_flow,
                                   "C3": C3_flow,
                                   "iC4": iC4_flow,
                                   "nC4": nC4_flow,
                                   "iC5": iC5_flow,
                                   "nC5": nC5_flow,
                                   "Molar concentration": molars,
                                   "C2_": C2_conc,
                                   "C3_": C3_conc,
                                   "iC4_": iC4_conc,
                                   "nC4_": nC4_conc,
                                   "iC5_": iC5_conc,
                                   "nC5_": nC5_conc,
                                   })

        df_streams = df_streams.sort_values(by=["Stream number"])
        df_streams = df_streams.T
    with pd.ExcelWriter(f'{name}.xlsx') as writer:
        df_col.to_excel(writer, sheet_name="ColumnTable")
        df_streams.to_excel(writer, sheet_name="StreamTable")
        writer.save()
        writer.close()

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
