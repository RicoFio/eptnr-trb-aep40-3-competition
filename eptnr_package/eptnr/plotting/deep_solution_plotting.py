from matplotlib import pyplot as plt
from typing import List, Dict


def plot_nn_loss_reward_epsilon(policy_net_loss: List[float], rewards_over_episodes: Dict[str, List[float]],
                                eps_values_over_steps: List[float], title: str,
                                fig: plt.Figure = None, ax: plt.Axes = None):
    """

    Args:
        policy_net_loss:
        rewards_over_episodes:
        eps_values_over_steps:
        title:
        fig:
        ax:

    Returns:

    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax[0].plot(range(len(policy_net_loss)), policy_net_loss, label='policy net loss')
    ax[0].legend()

    lns = []

    for rew_name, rewards in rewards_over_episodes.items():
        lns.extend(ax[1].plot(range(len(rewards)), rewards, label=rew_name, alpha=0.7))

    ax2 = ax[1].twinx()

    ax2.yaxis.label.set_color('purple')
    ax2.tick_params(axis='y', colors='purple')
    lns2 = ax2.plot(range(len(eps_values_over_steps)), eps_values_over_steps, color='purple', label='epsilon')
    lns.extend(lns2)

    # Generate legend. See https://stackoverflow.com/a/5487005 for reference
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc=3)

    ax[0].set_ylabel('MSE Loss')
    ax[1].set_ylabel('Reward')
    ax[1].set_xlabel('Episodes')
    ax2.set_ylabel('Epsilon')

    # fig.suptitle(title)

    fig.tight_layout()

    return fig, ax
