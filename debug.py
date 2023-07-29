import torch as th



def debug_mean_std(mean, log_std):

    min_mean = th.min(mean, dim=0).values
    max_mean = th.max(mean, dim=0).values
    mean_mean = th.mean(mean, dim=0)

    std = th.exp(log_std)
    min_std = th.min(std, dim=0).values
    max_std = th.max(std, dim=0).values
    mean_std = th.mean(std, dim=0)

    print(f"mean: \n{min_mean}\n{max_mean}\n{mean_mean}")
    print(f"std: \n{min_std}\n{max_std}\n{mean_std}")


def debug_action(sample_action):

    min_ac = th.min(sample_action, dim=0).values
    max_ac = th.max(sample_action, dim=0).values
    mean_ac = th.mean(sample_action, dim=0)

    # squshed_ac = th.tanh(sample_action)
    # min_squshed_ac = th.min(squshed_ac, dim=0).values
    # max_squshed_ac = th.max(squshed_ac, dim=0).values
    # mean_squshed_ac = th.mean(squshed_ac, dim=0)

    print(f"ac: \n{min_ac}\n{max_ac}\n{mean_ac}")
    # print(f"sqh_ac: \n{min_squshed_ac}\n{max_squshed_ac}\n{mean_squshed_ac}")
