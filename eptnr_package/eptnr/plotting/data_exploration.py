import pandas as pd
import seaborn as sns
from ..rewards.utils.graph_computation_utils import get_tt_hops_com_dfs


def plot_travel_time_histogram(graph, census, ax=None, min_x=None, max_x=None, min_y=None, max_y=None):
    tt_df, _, _ = get_tt_hops_com_dfs(graph, census, 50)
    temp_tt_df = tt_df.copy()
    categories = pd.Categorical(temp_tt_df['group'])
    temp_tt_df['group'] = categories
    temp_tt_df = temp_tt_df.pivot(columns='group')

    melted_temp_tt = temp_tt_df.melt()[['group', 'value']]
    melted_temp_tt.value = pd.to_numeric(melted_temp_tt.value)
    melted_temp_tt['travel time'] = melted_temp_tt.value

    if not ax:
        _, ax = plt.subplots()
    sns.histplot(
        melted_temp_tt,
        x='travel time',
        hue='group',
        multiple='dodge',
        kde=True,
        shrink=.75,
        bins=100,
        palette=categories.unique().to_list(),
        ax=ax
    )

    ax.vlines(
        melted_temp_tt.value.mean(),
        ymin=0,
        ymax=melted_temp_tt.groupby('group').value_counts().max(),
        linestyles="dashed",
        label="mean travel time"
    )

    if (min_x is not None) and (max_x is not None):
        ax.set_xlim(min_x, max_x)
    if (min_y is not None) and (max_y is not None):
        ax.set_ylim(min_y, max_y)
    return ax