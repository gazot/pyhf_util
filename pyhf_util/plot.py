import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

from .wrapper import df_stats

def plt_config(use_tex=False):
    """Customize plot layout."""
    sns.set_style('darkgrid')
    rcParams['font.family'] = ['serif']
    rcParams['font.serif'] = ['Ubuntu']
    rcParams.update({'font.monospace': ['Ubuntu Mono']})
    rcParams.update({'font.size': 12})
    rcParams.update({'axes.labelsize': 13})

    if use_tex:
        pass 


def savefig(path):
    plt.savefig(path, bbox_inches='tight')



# -------- DEFINE COLORS --------

ec_gray = (0.1, 0.1, 0.1, 0.5)

gray = (0.1, 0.1, 0.1, 0.2)
dgray = (0.1, 0.1, 0.1, 0.6)


dblue = (0.4, 0.4, 0.8, 0.8)
dred = (0.8, 0.2, 0.2, 0.8)
dgreen = (0.2, 0.5, 0.2, 0.6)

# seaborn bkg
bkg = (0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 1.0)



# -------- PLOT FUNCTIONS --------

def plot_lin_regression(ax, x, y, order: int=0, **kwargs):
    """Linear regression of data points (x, yi) while type(y) is a list."""
    for i in range(len(y)):
        p = np.polyfit(x, y[i], order)
        ax.plot(x, np.polyval(p, x), **kwargs)



def box_bar_plot(df, ax, yvalue, color, legend, loc='best', title=None, outlier=True):
    """Create a fance box bar plot."""

    # Bar plot 
    bar_df, _ = df_stats(df, yvalue)
    bar_args = dict(ec=bkg, color=color, lw=1.5)
    bar_df.plot(kind='barh', ax=ax, **bar_args)

    # Box plot 
    box_args = dict(width=0.5, showfliers=outlier, 
                    boxprops={'facecolor':gray, 'edgecolor': 'black', 'zorder':2}, 
                    whiskerprops={'color':'black'},
                    capprops={'color':'black'},
                    medianprops={'color': 'black'}, 
                    flierprops={'color': 'black', 'ms':3}, linewidth=1.5)
    sns.boxplot(y=yvalue, x='value', data=df, ax=ax, hue='Model', **box_args)

    # repair legend
    leg = ax.legend(legend, loc=loc, title=title)
    leg.legendHandles[0].set_color(color[0])
    leg.legendHandles[1].set_color(color[1])


