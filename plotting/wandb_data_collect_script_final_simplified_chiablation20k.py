import argparse
import os

import matplotlib.pyplot as plt
import neatplot
import numpy as np
from scipy.stats import sem
from visualisations_utils_wandb_api import download_runs, group_process_runs, process_max_fields, process_runs

neatplot.set_style()

# Constants and configurations
ENTITY = "robust-rl-project"
PROJECT = "bandits_dpo"
REWARD_FUNC = "swapped"  # in {'swapped', 'flipped', 'same'}
SETTINGS = {
    "uneven_imbalanced_ipo": [
        (
            f"state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_type{REWARD_FUNC}eval_metricargmax_state-1",
            "uneven_imbalanced_ipo",
        ),
        (
            f"state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_type{REWARD_FUNC}eval_metricargmax_chiablation",
            "chi_ripo",
        ),
    ],
    "uneven_imbalanced_dpo": [
        # (f'state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_type{REWARD_FUNC}eval_metricargmax_iason_uneven_imbal_osc_dpo', 'DPO'),
        # (f'state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_type{REWARD_FUNC}eval_metricargmax_iason_uneven_imbal_osc_imp', 'IS-DPO'),
        (
            f"state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_type{REWARD_FUNC}eval_metricargmax_chiablation_isdpo20k",
            "IS-DPO",
        ),
        (
            f"state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_type{REWARD_FUNC}eval_metricargmax_chiablation_rdpo20k",
            "chi_rdpo",
        ),
        # (f'state_dim2action_num8group_num2pref_data_num300weights[0.2,0.8]feature_type{REWARD_FUNC}eval_metricargmax_chiablation_rdpo', 'chi_rdpo'),
    ],
}
ALGORITHMS = {
    "ripo_th": "Robust IPO Theory",
    "ripo_pr": "Robust IPO Prac",
    "ipo": "IPO",
    "imp_samp": "Importance_sampling",
}

pref_data_num = 300


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="even_imbalanced_dpo")
    # convention X_Y_Z: X={'even','uneven'}, Y={'balanced','imbalanced'}, Z={'dpo','ipo','all'}
    return parser.parse_args()


def get_setting_details(setting_key: str):
    if "all" in setting_key:
        assert setting_key[-3:] == "all", "Wrong setting_key convention."
        dpo_key = setting_key[:-3] + "dpo"
        ipo_key = setting_key[:-3] + "ipo"
        group_list = SETTINGS[ipo_key]
        group_list_2 = SETTINGS[dpo_key]
        group_list.extend(group_list_2)
    else:
        group_list = SETTINGS[setting_key]
    weights_array = np.array(group_list[0][0].split("weights[")[-1].split("]")[0].split(","), dtype=float)
    pref_data_num = group_list[0][0].split("pref_data_num")[1].split("weights")[0]
    return group_list, weights_array, pref_data_num


def create_filter_dicts(groups: list[tuple[str, str]], uneven: bool):
    base_filter_ipo = {
        "config.ipo_grad_type": "linear",
        "config.reg_coef": 0.1,
        "config.dpo_type": "rdpo",
        "State": "finished",
        "config.lamba": 0,
        "config.use_uneven_grp": uneven,
    }
    base_filter_dpo = {"State": "finished", "config.use_uneven_grp": uneven}

    filters = []
    group_names = []

    print("FILTER GROUPS = ", groups)

    for group in groups:
        if "chi" in group[1].lower():
            if "ripo" in group[1]:
                for chi in list(map(lambda x: x / 10, range(0, 12, 2))):
                    chi_filter = {**base_filter_ipo, "group": group[0], "config.dpo_type": "rdpo", "config.chi": chi}
                    filters.append(chi_filter)
                    if chi == 1:
                        group_names.append("$\\textbf{GR-IPO}$" + rf" $\chi={chi}$")
                    elif chi == 0:
                        group_names.append("GR-IPO" + rf" $\chi={chi}$" + " (IS-IPO)")
                    else:
                        group_names.append("GR-IPO" + rf" $\chi={chi}$")
            else:
                for chi in list(map(lambda x: x / 10, range(0, 12, 2))):
                    chi_filter = {**base_filter_dpo, "group": group[0], "config.dpo_type": "rdpo", "config.chi": chi}
                    filters.append(chi_filter)
                    if chi == 1:
                        group_names.append("$\\textbf{GR-DPO}$" + rf" $\chi={chi}$")
                    else:
                        group_names.append("GR-DPO" + rf" $\chi={chi}$")
        elif "ipo" in group[1].lower():
            dpo_filter = {
                **base_filter_ipo,
                "group": groups[0][0],
                "config.importance_sampling": True,
                "config.importance_sampling_weights": {"$in": ["0.5,0.5"]},
                "config.dpo_num_iters": 10,
            }
            imp_samp_filter = {
                **base_filter_ipo,
                "group": groups[0][0],
                "config.importance_sampling": True,
                "config.importance_sampling_weights": {"$nin": ["0.5,0.5"]},
                "config.dpo_num_iters": 10,
            }
            theory_filter = {
                **base_filter_ipo,
                "group": groups[0][0],
                "config.importance_sampling": False,
                "config.rdpo_exp_step_size": 0.01,
                "config.use_theory": True,
                "config.dpo_num_iters": 100,
            }
            # avg_filter = {**base_filter_ipo, 'group': groups[0][0], 'config.importance_sampling': False, 'config.rdpo_exp_step_size': 0.01, 'config.use_theory': False, 'config.dpo_num_iters': 100}
            continue  ## no IPO/DPO

            if "uneven_balanced" in group[1]:
                filters.extend([dpo_filter, theory_filter])
                group_names.extend(["IPO", "GR-IPO"])
            else:
                filters.extend([dpo_filter])  # , imp_samp_filter, theory_filter
                group_names.extend(["IPO"])  # , 'IS-IPO', 'GR-IPO'
            continue
        elif "dpo" in group[1].lower():
            filter = {
                **base_filter_dpo,
                "group": group[0],
                "config.ipo_grad_type": "justdpo",
                "config.dpo_type": "rdpo" if ("rdpo" in group[0] or "isdpo" in group[0]) else "dpo",
                "config.importance_sampling": "isdpo" in group[0],
                "config.importance_sampling_weights": {"$nin": ["0.5,0.5"]},
                "config.use_theory": False,
                "config.chi": 1,
            }
            group_names.append(group[1])
            # print(f'FILTERS: {filter}\n')
            filters.append(filter)
        else:
            print("GROUP 1 = ", group[1])
            raise ValueError

    print("FILTERS: ", filters, group_names)
    return filters, group_names


def determine_algorithm(filters_dict):
    if filters_dict["config.ipo_grad_type"] == "linear":  # IPO
        if not filters_dict["config.importance_sampling"]:
            return "GR-IPO"
        if filters_dict["config.importance_sampling_weights"] == {"$in": ["0.5,0.5"]}:
            return "IPO"
        return "IS-IPO"

    if filters_dict["config.importance_sampling"] == True:
        return "IS-DPO"
    if filters_dict["config.dpo_type"] == "dpo":
        return "DPO"
    return "GR-DPO"


def prepare_metric_data(
    filters_dicts, group_names, metrics, all_avg_metrics_at_iterations, all_sem_metrics_at_iterations, metric_titles
):
    metric_values = []
    metric_sem = []
    labels = []
    for metric_name in metrics:
        for i, filters_dict in enumerate(filters_dicts):
            if group_names is not None:
                algo = group_names[i]
            else:
                algo = determine_algorithm(filters_dict)
            data_num = pref_data_num  # assuming this is predefined somewhere
            avg = all_avg_metrics_at_iterations[metric_name][i]
            sem = all_sem_metrics_at_iterations[metric_name][i]
            # name=metric_titles[metric_name]
            metric_values.append(avg)
            metric_sem.append(sem)
            labels.append(algo)
    return metric_values, metric_sem, labels


def plot_metric_with_error_bands(
    iteration_index,
    metric_values,
    metric_sem,
    labels,
    plot_title,
    subfolder_path,
    file_name,
    metric,
    colors=None,
    extend=False,
):
    plt.figure(figsize=(12, 6))
    # for i, (avg, sem) in enumerate(zip(metric_values, metric_sem)):

    avg_is = []
    errp_is = []
    errn_is = []

    for avg, sem, label in zip(metric_values, metric_sem, labels):
        if extend and len(avg) != len(iteration_index):
            avg = np.append(avg, [avg[-1]] * (len(iteration_index) - len(avg)))
            sem = np.append(sem, [sem[-1]] * (len(iteration_index) - len(sem)))
        # color = colors[i] if colors else None

        errp = avg + sem
        errn = avg - sem

        # if metric=='val_loss' and label=='DPO':
        #    continue

        if (metric == "val_loss") and ("GR-DPO" in label or "$\\textbf{GR-DPO}$" in label or "IS-DPO" in label):
            avg *= 300
            errp *= 300
            errn *= 300

        #    avg = np.log10(avg)
        #    errp = np.log10(avg + sem)
        #    errn = np.log10(avg - sem)

        if plot_title == "Average Validation Group Loss":
            avg_is.append(avg)
            errp_is.append(errp)
            errn_is.append(errn)
            continue

        plt.plot(iteration_index, avg, label=label, linewidth=2)
        plt.fill_between(iteration_index, errn, errp, alpha=0.2)

    if plot_title == "Average Validation Group Loss":
        for i in range(int(len(metric_values) / 2)):
            avg = (avg_is[i] + avg_is[i + int(len(metric_values) / 2)]) * 0.5
            errp = (errp_is[i] + errp_is[i + int(len(metric_values) / 2)]) * 0.5
            errn = (errn_is[i] + errn_is[i + int(len(metric_values) / 2)]) * 0.5
            label = labels[i]

            plt.plot(iteration_index, avg, label=label, linewidth=2)
            plt.fill_between(iteration_index, errn, errp, alpha=0.2)

    plt.grid(visible=True, linewidth=2)

    plt.tick_params(axis="both", which="major", labelsize=35)
    plt.tick_params(axis="both", which="minor", labelsize=35)

    plt.title(plot_title, fontsize=55)
    plt.xlabel("Iterations", fontsize=50)
    plt.ylabel("Value", fontsize=50)
    plt.legend(fontsize=20, loc="center right")
    neatplot.save_figure(f"{subfolder_path}/{REWARD_FUNC}_{file_name}", ext_list="pdf")
    plt.close()


def plot_metric_bars(
    metric_config,
    filters_dicts,
    group_names,
    subfolder_path,
    all_avg_metrics_at_iterations,
    all_sem_metrics_at_iterations,
    weights_array,
):
    plt.figure(figsize=(12, 6))
    legend_show = True
    all_algos = []
    for i, filters_dict in enumerate(filters_dicts):
        if group_names is not None:
            algo = group_names[i]
        else:
            algo = determine_algorithm(filters_dict)

        all_algos.append(algo)
        data_num = pref_data_num
        metrics_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metric_config["metrics"]]
        metrics_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metric_config["metrics"]]

        bar_width = 0.1 if "group_loss" in metric_config["metrics"][0] else 0.2
        offset = i * bar_width
        positions = np.arange(len(metrics_end_avg)) + offset

        plt.bar(
            positions,
            height=metrics_end_avg,
            yerr=metrics_end_sem,
            width=bar_width * 0.95,
            capsize=5,
            alpha=0.7,
            label=f"{algo}",
        )
        # plt.gca().invert_yaxis()

    if "Max" not in metric_config["title"]:
        plt.xticks(positions, [f"Group {i+1} Ratio {weights_array[i]}" for i in range(len(metrics_end_avg))])
    else:
        plt.xticks([i * bar_width for i in range(len(filters_dicts))], all_algos)
        legend_show = False

    plt.tick_params(axis="x", which="major", labelsize=25)
    plt.tick_params(axis="y", which="major", labelsize=35)
    plt.tick_params(axis="both", which="minor", labelsize=35)

    plt.title(metric_config["title"], fontsize=55)
    plt.ylabel("Value", fontsize=50)
    plt.xlabel("Methods", fontsize=50)

    if legend_show is True:
        plt.legend(fontsize=20)
    neatplot.save_figure(f'{subfolder_path}/{REWARD_FUNC}_{metric_config["file_suffix"]}', ext_list="pdf")
    plt.close()


def main(args):
    setting = "uneven_imbalanced_dpo"  # args.setting

    groups, weights_array, pref_data_num = get_setting_details(setting)
    filters_dicts, group_names = create_filter_dicts(groups, "uneven" in setting)

    # metrics_to_collect = ['grad_norm', 'train_loss', 'reward_err_1', 'reward_err_2', 'reward_param_1', 'reward_param_2', 'reward_param_3', 'reward_param_4','group_weight_1','group_weight_2','val_loss','train_group_loss_1','train_group_loss_2','val_group_loss_1','val_group_loss_2','hist_group_loss_1','hist_group_loss_2','max_val_grp_loss','max_train_grp_loss','max_reward_err','max_kl_dist']
    metrics_to_collect = ["val_loss", "max_val_grp_loss", "val_group_loss_1", "val_group_loss_2"]
    all_metrics_history = {metric: [] for metric in metrics_to_collect}

    all_runs = []
    # Loop through each filters_dict value
    for filters_dict in filters_dicts:
        # Download runs for the current filters_dict
        runs = download_runs(ENTITY, PROJECT, filters_dict)
        all_runs.append(runs)
        # print(len(runs))
        metrics_history = {}

        for metric in metrics_to_collect:
            algo = filters_dict["config.dpo_type"]
            data_num = pref_data_num
            if algo == "dpo" and "group_weight" in metric:
                metrics_history[metric] = []
                continue
            metrics_history[metric] = process_runs(runs, field=metric, time_field="Iteration")

        # Accumulate metrics data for each configuration
        for metric in metrics_to_collect:
            all_metrics_history[metric].append(metrics_history[metric])

    iteration_len = 0
    iteration_index = 0
    for runs in all_runs:
        for run in runs:
            iteration_index_1 = run[["Iteration"]].dropna().values.ravel()
            # print(iteration_index_1)
            if len(iteration_index_1) > iteration_len:
                iteration_len = len(iteration_index_1)
                iteration_index = iteration_index_1

    base_folder = "bandit-dpo-plots-final-chi-v6"
    os.makedirs(base_folder, exist_ok=True)
    subfolder_name = (
        f"{len(filters_dicts)}_setting_{setting}"  # f"{filters_dicts[0]['config.dpo_type']}{len(filters_dicts)}_v2"
    )
    subfolder_path = os.path.join(base_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    all_avg_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}
    all_sem_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}

    for i, filters_dict in enumerate(filters_dicts):
        for metric in metrics_to_collect:
            values_matrix = all_metrics_history[metric][i]
            for j in range(len(values_matrix)):
                values_matrix[j].dropna(inplace=True)

            if len(values_matrix) == 0:  # group_weight in DPO group is empty
                avg_values = np.float64(np.nan)
                sem_values = np.float64(np.nan)
            else:
                avg_values = np.mean(values_matrix, axis=0)
                sem_values = sem(all_metrics_history[metric][i], axis=0)
            all_avg_metrics_at_iterations[metric].append(avg_values.ravel())
            all_sem_metrics_at_iterations[metric].append(sem_values.ravel())

    # Plotting configurations
    plot_configs = [
        # ('grad_norm',),
        # ('train_loss',),
        ("val_loss",),
        # ('reward_err_1', 'reward_err_2'),
        # ('reward_param_1', 'reward_param_2'),
        # ('reward_param_3', 'reward_param_4'),
        # ('group_weight_1', 'group_weight_2'),
        # ('train_group_loss_1', 'train_group_loss_2'),
        ("val_group_loss_1", "val_group_loss_2"),
        # ('hist_group_loss_1', 'hist_group_loss_2'),
        ("max_val_grp_loss",),
        # ('max_train_grp_loss',),
        # ('max_reward_err',),
        # ('max_kl_dist',)
    ]
    titles_dict = {
        "grad_norm": "Gradient Norm",
        "train_loss": "Training Loss",
        "val_loss": "Average Validation Loss",
        "reward_err_1_reward_err_2": "Reward Errors",
        "reward_param_1_reward_param_2": "Reward Parameters 1 and 2",
        "reward_param_3_reward_param_4": "Reward Parameters 3 and 4",
        "group_weight_1_group_weight_2": "Group Weights",
        "train_group_loss_1_train_group_loss_2": "Training Group Losses",
        "val_group_loss_1_val_group_loss_2": "Average Validation Group Loss",
        "hist_group_loss_1_hist_group_loss_2": "Historical Group Losses",
        "max_val_grp_loss": "Max Validation Group Loss",
        "max_train_grp_loss": "Max Training Group Loss",
        "max_reward_err": "Max Reward Error",
        "max_kl_dist": "Max KL Distance",
    }

    metrics_titles = {
        "grad_norm": "Gradient Norm",
        "train_loss": "Training Loss",
        "val_loss": "Average Validation Loss",
        "reward_err_1": "Reward Error Group 1",
        "reward_err_2": "Reward Error Group 2",
        "reward_param_1": "Reward Parameter 1",
        "reward_param_2": "Reward Parameter 2",
        "reward_param_3": "Reward Parameter 3",
        "reward_param_4": "Reward Parameter 4",
        "group_weight_1": "Group Weight 1",
        "group_weight_2": "Group Weight 2",
        "val_loss": "Validation Loss",
        "train_group_loss_1": "Training Group Loss 1",
        "train_group_loss_2": "Training Group Loss 2",
        "val_group_loss_1": "Validation Group Loss 1",
        "val_group_loss_2": "Validation Group Loss 2",
        "hist_group_loss_1": "Historical Group Loss 1",
        "hist_group_loss_2": "Historical Group Loss 2",
        "max_val_grp_loss": "Maximum Validation Group Loss",
        "max_train_grp_loss": "Maximum Training Group Loss",
        "max_reward_err": "Maximum Reward Error",
        "max_kl_dist": "Maximum KL Distance",
    }

    for metrics in plot_configs:
        values, sems, labels = prepare_metric_data(
            filters_dicts,
            group_names,
            metrics,
            all_avg_metrics_at_iterations,
            all_sem_metrics_at_iterations,
            metrics_titles,
        )
        metric_name = "_".join(metrics)

        # print(f'AVG max_rew_err at iter: ', all_avg_metrics_at_iterations['max_reward_err'])
        # print('\n')
        # print(f'VALUES at iter: ', values)
        # print('\n\n')

        plot_metric_with_error_bands(
            iteration_index,
            values,
            sems,
            labels,
            f"{titles_dict[metric_name]}",
            subfolder_path,
            f"{metric_name}",
            metric=metric_name,
            extend=True,
        )

    return

    # Define a list of metric configurations for each plot
    metrics_configs = [
        {
            "metrics": [metric for metric in metrics_to_collect if "reward_err_" in metric],
            "title": "Converged Reward Errors",
            "file_suffix": "reward_bars",
        },
        {
            "metrics": [metric for metric in metrics_to_collect if "train_group_loss" in metric],
            "title": "Converged Group Train Loss",
            "file_suffix": "train_group_loss_bars",
        },
        {
            "metrics": [metric for metric in metrics_to_collect if "val_group_loss" in metric],
            "title": "Converged Group Validation Loss",
            "file_suffix": "val_group_loss_bars",
        },
        {
            "metrics": [metric for metric in metrics_to_collect if "max_reward_err" in metric],
            "title": "Converged Max Reward Error",
            "file_suffix": "max_reward_bars",
        },
        {
            "metrics": [metric for metric in metrics_to_collect if "max_train_grp_loss" in metric],
            "title": "Converged Max Group Train Loss",
            "file_suffix": "max_train_group_loss_bars",
        },
        {
            "metrics": [metric for metric in metrics_to_collect if "max_val_grp_loss" in metric],
            "title": "Converged Max Group Validation Loss",
            "file_suffix": "max_val_group_loss_bars",
        },
        {
            "metrics": [metric for metric in metrics_to_collect if "max_kl_dist" in metric],
            "title": "Converged Max KL Distance",
            "file_suffix": "max_kl_distance_bars",
        },
    ]

    # Loop through each configuration and plot
    for config in metrics_configs:
        plot_metric_bars(
            config,
            filters_dicts,
            group_names,
            subfolder_path,
            all_avg_metrics_at_iterations,
            all_sem_metrics_at_iterations,
            weights_array,
        )


if __name__ == "__main__":
    main(parse_args())
