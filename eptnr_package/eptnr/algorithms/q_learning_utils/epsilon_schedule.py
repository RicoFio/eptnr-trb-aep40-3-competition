import numpy as np
from matplotlib import pyplot as plt
import math
from typing import (
    Iterable,
    Union,
    Tuple,
)


class EpsilonSchedule:

    def __init__(self, eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 200,
                 static_eps_steps: int = 0) -> None:
        """

        Args:
            eps_start:
            eps_end:
            eps_decay:
            static_eps_steps: [Optional] Number of steps eps_start should be held for.
        """
        assert 1 >= eps_start > 0
        assert 1 >= eps_end >= 0
        assert eps_decay > 0

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.static_eps_steps = static_eps_steps

        self.steps_done = 0

    @property
    def curr_eps(self):
        return self._compute_eps(self.steps_done)

    def _compute_eps(self, step: int) -> np.array:
        if step < self.static_eps_steps:
            return self.eps_start
        elif step >= self.static_eps_steps:
            step -= self.static_eps_steps
            return self.eps_end + (self.eps_start - self.eps_end) * \
                   math.exp(-1. * step / self.eps_decay)

    def make_step(self):
        self.steps_done += 1

    def get_current_eps(self):
        return self.curr_eps

    def plot_schedule(self, figsize: Tuple[int, int] = (5, 5)) -> Tuple[plt.Figure, plt.Axes]:
        # To calculate at which step eps == self.eps_end we have:
        # steps = eps_decay * ( ln(eps_start - eps_end) - 1)
        # Add the amount of steps for which eps is held static and some margin (10 steps) at the end
        max_steps = round(self.static_eps_steps + self.eps_decay * (
                    math.log(self.eps_start - self.eps_end) - math.log(self.eps_end)) + 10)
        steps = np.arange(0, max_steps, 1)
        epsilons = [self._compute_eps(s) for s in steps]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(steps, epsilons)
        if self.static_eps_steps > 0:
            ax.axvline(x=self.static_eps_steps)

        return fig, ax
