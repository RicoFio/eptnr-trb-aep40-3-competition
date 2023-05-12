from dataclasses import dataclass
import numpy as np


@dataclass
class PartialReward:
    min_value: float
    max_value: float
    reward_value: float

    @property
    def max_diff_min(self):
        return self.max_value - self.min_value

    @property
    def max_diff_actual(self):
        return self.max_value - self.reward_value

    @property
    def ma_d_ac_div_ma_d_mi(self):
        return np.divide(self.max_diff_actual, self.max_diff_min)


@dataclass
class PartialRewardGenerator:
    min_value: float
    max_value: float

    def generate_reward(self, reward_value: float):
        return PartialReward(self.min_value, self.max_value, reward_value)


def chebyshev_weight_computation(*fs: PartialReward):
    numerators = [f.ma_d_ac_div_ma_d_mi for f in fs]
    denominator = np.sum([f.ma_d_ac_div_ma_d_mi for f in fs])

    if denominator != 0:
        return np.divide(numerators, denominator)
    else:
        max_reward = max([f.reward_value for f in fs])
        return [1 if f.reward_value == max_reward else 0 for f in fs]


def chebyshev_reward_computation(*fs: PartialReward):
    weights = chebyshev_weight_computation(*fs)
    assert np.isclose(np.sum(weights), 1)
    return np.sum([w * f.reward_value for w, f in zip(weights, fs)])
