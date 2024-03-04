import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
import ast

from algos.linear_bandit.mle import MLERewardLearning
from algos.linear_bandit.pg import PolicyGradient
from algos.linear_bandit.dpo import DirectPolicyOptimization
from algos.linear_bandit.group_dpo import GroupDirectPolicyOptimization
from envs.linear_bandit import LinearBandit, ret_feature_func
from envs.group_linear_bandit import GroupLinearBanditSep, GroupLinearBandit, ret_feature_func
from utils.io_utils import save_code, save_config, create_log_dir
from utils.logger import Logger
from utils.collect_data import (
    ret_uniform_policy_group,
    collect_preference_data,
    collect_group_preference_data,
    collect_rl_data,
    merge_datasets,
    pref_to_rl,
)
from utils.utils import return_apt_weights

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="group_linear_bandit")
    parser.add_argument("--state_dim", type=int, default=1)
    parser.add_argument("--action_num", type=int, default=4)
    parser.add_argument("--group_num", type=int, default=2)
    parser.add_argument("--agent", type=str, default="pg")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--logdir", type=str, default="log")
    #parser.add_argument("--flip_feature", action="store_true")

    parser.add_argument("--pref_data_num", type=int, default=500)
    parser.add_argument('--weights',type=str,default='equal')
    parser.add_argument("--val_data_num", type=int, default=50)
    parser.add_argument('--val_weights',type=str,default='equal')
    parser.add_argument("--num_trials_for_eval", type=int, default=1000)
    parser.add_argument('--test_weights',type=str,default='equal')

    parser.add_argument("--mle_num_iters", type=int, default=100)
    parser.add_argument("--mle_adaptive", action="store_true")
    parser.add_argument("--mle_ada_coef", type=float, default=1.0)
    parser.add_argument("--mle_step_size", type=float, default=0.1)

    parser.add_argument("--rl_data_ratio", type=float, default=4)
    parser.add_argument("--reg_coef", type=float, default=1.0)

    parser.add_argument("--dpo_num_iters", type=int, default=200)
    parser.add_argument("--dpo_adaptive", action="store_true")
    parser.add_argument("--dpo_ada_coef", type=float, default=1.0)
    parser.add_argument("--dpo_step_size", type=float, default=0.1)

    parser.add_argument("--pg_num_iters", type=int, default=1000)
    parser.add_argument("--pg_adaptive", action="store_true")
    parser.add_argument("--pg_ada_coef", type=float, default=1.0)
    parser.add_argument("--pg_step_size", type=float, default=0.1)

    return parser.parse_args()


def get_reward_func(reward_param: np.ndarray, feature_func):
    def reward_func(state, action, group_id):
        feature = feature_func(state, action, group_id)
        rew = np.dot(feature, reward_param)

        return rew

    return reward_func


def main(args):
    np.random.seed(args.seed)
    log_dir = create_log_dir(args)
    save_code(log_dir)
    save_config(args.__dict__, log_dir)

    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir)

    state_dim = args.state_dim
    action_num = args.action_num
    group_num = args.group_num

    feature_dim = 2 * args.state_dim
    num_trials_for_eval = args.num_trials_for_eval
    feature_func = ret_feature_func(num_action=action_num, state_dim=state_dim, group_num=group_num)
    # reward_param = np.random.standard_normal(feature_dim)
    # reward_param = np.array([2.0, 1.0, 1.0, 2.0], np.float32)
    reward_param = np.array([[1.0, 2.0],[2.0,1.0]], np.float32)

    assert group_num == np.shape(reward_param)[0], "The feature is invalid."

    # reward_param /= np.sqrt(np.sum(np.square(reward_param)))
    env = GroupLinearBanditSep(
        state_dim,
        action_num,
        group_num,
        reward_param,
        feature_func,
        num_trials_for_eval=num_trials_for_eval,
    )

    weights,val_weights,test_weights=return_apt_weights(args.weights,group_num),return_apt_weights(args.val_weights,group_num),return_apt_weights(args.test_weights,group_num)
    
    opt_policy = env.get_opt_policy()
    uniform_policy = ret_uniform_policy_group(action_num)
    #Generate datasets:
    pref_data = collect_group_preference_data(args.pref_data_num, env, weights, uniform_policy)
    val_pref = collect_group_preference_data(args.val_data_num, env, val_weights, uniform_policy)
    test_pref = collect_group_preference_data(args.num_trials_for_eval, env, test_weights, uniform_policy)


    
    opt_reward = env.evaluate_reward_group_wise(policy=opt_policy,states=test_pref)

    unif_policy_rew = env.evaluate_reward_group_wise(policy=uniform_policy,states=test_pref)

    formatted_opt_reward = ", ".join([f"{reward:.4f}" for reward in opt_reward])
    formatted_unif_policy_rew = ", ".join([f"{reward:.4f}" for reward in unif_policy_rew])
    logger.info(
        f"optimal policy reward: {formatted_opt_reward}, uniform policy reward: {unif_policy_rew}."
    )

    # learn the reward function
    reward_model = MLERewardLearning(
        feature_func,
        feature_dim,
        args.mle_step_size,
        args.mle_num_iters,
        args.mle_adaptive,
        args.mle_ada_coef,
    )
    loss, l2_dist, acc = reward_model.train_by_cvxpy_group(
        dataset=pref_data, true_reward_param=reward_param
    )
    logger.info(f"Reward loss: {loss:.4f}, l2 distance: {l2_dist:.4f}, acc: {acc:.2f}.")

    learned_reward_func = reward_model.get_reward_func
    learned_reward_param = reward_model.get_reward_param
    logger.info("True reward parameter: {}".format(reward_param))
    logger.info("Learned reward parameter: {}".format(learned_reward_param))

    # Oracle test
    learned_env = GroupLinearBandit(
        state_dim,
        action_num,
        group_num,
        learned_reward_param,
        feature_func,
        num_trials_for_eval=num_trials_for_eval,
    )
    learned_oracle_opt_policy = learned_env.get_opt_policy()
    learned_oracle_opt_reward = env.evaluate_reward_group_wise(policy=learned_oracle_opt_policy,states=test_pref)

    formatted_learned_oracle_opt_reward  = ", ".join([f"{reward:.4f}" for reward in learned_oracle_opt_reward])
    logger.info(f"Learned oracle reward: {formatted_learned_oracle_opt_reward}")

    # Train the RL on the preference data
    logger.info(f"Train a policy solely on the preference data (DPO).")
    # learn the policy
    policy_feature_func = ret_feature_func(
        num_action=action_num, state_dim=state_dim, group_num=group_num
    )
    agent = GroupDirectPolicyOptimization(
        state_dim=state_dim,
        action_num=action_num,
        group_num=group_num,
        feature_dim=feature_dim,
        feature_func=policy_feature_func,
        ref_policy=uniform_policy,
        reg_coef=args.reg_coef,
        step_size=args.dpo_step_size,
        num_iters=args.dpo_num_iters,
        is_adaptive=args.dpo_adaptive,
        ada_coef=args.dpo_ada_coef,
        logger=logger,
    )

    # reward = agent.train_by_cvxpy(dataset=pref_data, env=env)
    reward = agent.train(dataset=pref_data, val_dataset=val_pref,test_dataset=test_pref, env=env)
    formatted_reward  = ", ".join([f"{reward:.4f}" for reward in reward])
    rew_error = [float((a-b)*100/a) for a,b in zip(opt_reward,reward)]
    formatted_rew_error  = ", ".join([f"{reward:.4f}" for reward in rew_error])
    policy_param = agent.get_param
    logger.info(
        f"Policy parameter learned solely on the preference data (DPO): {policy_param}."
    )
    logger.info(
        f"Training solely on the preference data (DPO), dataset size: {len(pref_data): d}, optimal reward: {formatted_opt_reward}, reward: {formatted_reward}, reward error percent: {formatted_rew_error}."
    )
    rew_err_dict, rew_dict = dict(), dict()
    rew_err_dict[args.pref_data_num] = rew_error
    rew_dict[args.pref_data_num] = reward
    save_path = os.path.join(log_dir, "reward_error_dpo.yml")
    yaml.dump(rew_err_dict, open(save_path, "w"), default_flow_style=False)
    save_path = os.path.join(log_dir, "reward_dpo.yml")
    yaml.dump(rew_dict, open(save_path, "w"), default_flow_style=False)

    """
    # RMB-PO
    logger.info(
        f"Train a policy on the preference data with policy-generated data (RMB-PO)."
    )
    rl_data = pref_to_rl(pref_data)
    agent = PolicyGradient(
        policy_feature_func,
        learned_reward_func,
        uniform_policy,
        feature_dim,
        action_num,
        args.reg_coef,
        args.pg_step_size,
        args.pg_num_iters,
        args.pg_adaptive,
        args.pg_ada_coef,
        logger=logger,
    )

    reward = agent.train(dataset=rl_data, env=env, learned_env=learned_env)
    rew_error = float(opt_reward - reward)
    policy_param = agent.get_param
    logger.info(f"Policy parameter (RMB-PO): {policy_param}.")
    logger.info(
        f"Training solely on the preference data (RMB-PO), dataset size: {len(rl_data): d}, optimal reward: {opt_reward: .4f}, reward: {reward: .4f}, reward error: {rew_error: .4f}."
    )
    rew_err_dict, rew_dict = dict(), dict()
    rew_err_dict[args.pref_data_num] = rew_error
    rew_dict[args.pref_data_num] = float(reward)
    save_path = os.path.join(log_dir, "reward_error_rmb_po.yml")
    yaml.dump(rew_err_dict, open(save_path, "w"), default_flow_style=False)
    save_path = os.path.join(log_dir, "reward_rmb_po.yml")
    yaml.dump(rew_dict, open(save_path, "w"), default_flow_style=False)

    # RMB-PO+: Collect a new RL data
    logger.info(f"Train a policy on the augmented data (RMB-PO+).")
    new_rl_data_num = int(args.pref_data_num * args.rl_data_ratio)
    new_rl_data = collect_rl_data(new_rl_data_num, env)
    aug_rl_data = merge_datasets(pref_data, new_rl_data)
    agent = PolicyGradient(
        policy_feature_func,
        learned_reward_func,
        uniform_policy,
        feature_dim,
        action_num,
        args.reg_coef,
        args.pg_step_size,
        args.pg_num_iters,
        args.pg_adaptive,
        args.pg_ada_coef,
        logger=logger,
    )
    reward = agent.train(aug_rl_data, env, learned_env)
    policy_param = agent.get_param
    logger.info(
        f"Policy parameter learned on the augmented data (RMB-PO+): {policy_param}."
    )
    rew_error = float(opt_reward - reward)
    logger.info(
        f"Training on the augmented data (RMB-PO+), augmented dataset size: {len(aug_rl_data): d}, optimal reward: {opt_reward: .4f}, reward: {reward: .4f}, reward error: {rew_error: .4f}."
    )
    rew_err_dict, rew_dict = dict(), dict()
    rew_err_dict[args.pref_data_num] = rew_error
    rew_dict[args.pref_data_num] = float(reward)
    save_path = os.path.join(log_dir, "reward_error_rmb_po_plus.yml")
    yaml.dump(rew_err_dict, open(save_path, "w"), default_flow_style=False)
    save_path = os.path.join(log_dir, "reward_aug_rmb_po_plus.yml")
    yaml.dump(rew_dict, open(save_path, "w"), default_flow_style=False)
    """

if __name__ == "__main__":
    main(parse_args())
