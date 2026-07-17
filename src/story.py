

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import ks_2samp
    return json, ks_2samp, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# A. Reply Network""")
    return


@app.cell
def _(json, pd):
    bsky_df = pd.read_csv("../data/bsky_reply_stats.csv")
    ts_df = pd.read_csv("../data/ts_reply_stats.csv")

    # Add parisenship
    with open('../data/ts_post_to_label.json', 'r') as f:
        ts_post_to_label = json.load(f)

    with open('../data/bsky_post_to_label.json', 'r') as f:
        bsky_post_to_label = json.load(f)


    for i, row in ts_df.iterrows():
        try:
            ts_df.loc[i, 'partisanship'] = ts_post_to_label[row['post']]
        except:
            ts_df.loc[i, 'partisanship'] = 'center'

    for i, row in bsky_df.iterrows():
        try:
            bsky_df.loc[i, 'partisanship'] = bsky_post_to_label[row['post']]
        except:
            bsky_df.loc[i, 'partisanship'] = 'center'
    ts_df['partisanship'] = ts_df['partisanship'].replace({'lean left': 'left', 'lean right': 'right'})
    bsky_df['partisanship'] = bsky_df['partisanship'].replace({'lean left': 'left', 'lean right': 'right'})

    ts_df.rename(columns={"topic": "topic_label"}, inplace=True)
    #import ts outliers
    ts_outliers = pd.read_csv("../data/ts_follower_outliers.csv")
    ts_df['outlier'] = False
    for i, row in ts_df.iterrows():
        if row['index'] in ts_outliers['post_id'].values:
            ts_df.loc[i, 'outlier'] = True

    df = pd.concat([ts_df, bsky_df], ignore_index=True)
    return bsky_df, bsky_post_to_label, df, ts_df, ts_post_to_label


@app.cell
def _(mo):

    # Widgets
    min_size = mo.ui.slider(1, 100, value=2, label="Minimum Size")
    platform_radio = mo.ui.radio(['bsky', 'ts'], label="Platform")
    return min_size, platform_radio


@app.cell
def _(df, min_size, mo, platform_radio, sns):
    sns.set_theme(rc={'figure.figsize':(11.7,5.27)})
    # Stack with widgets and simple plot
    mo.vstack([
        min_size,
        platform_radio,
        sns.histplot(df.loc[(df['size']>=min_size.value) & (df['platform']==platform_radio.value), "alignment_ratio"], bins=50, kde=True, color='blue', alpha=0.5, )
    ])


    return


@app.cell
def _(bsky_df, pd, ts_df):
    columns_to_keep = ['platform', 'topic_label', 'max_depth', 'size', 'breadth', 'index', 'structural_virality', 'partisanship', 'alignment_ratio']
    bsky_df_1 = bsky_df[columns_to_keep].reset_index(drop=True)
    ts_df_1 = ts_df[columns_to_keep + ['outlier']].reset_index(drop=True)
    df_1 = pd.concat([bsky_df_1, ts_df_1], ignore_index=True)
    df_1 = df_1.dropna(subset=['topic_label'])
    df_1['partisanship'] = df_1['partisanship'].replace({'error': 'center'})
    metrics = ['max_depth', 'size', 'breadth', 'structural_virality']
    return bsky_df_1, columns_to_keep, metrics, ts_df_1


@app.cell
def _(df):
    df['topic_label'].replace({'MAGA and Pro-Trump Hashtags and Advocacy': 'Pro-Trump and MAGA Advocacy'}, inpalce=True)
    return


@app.cell
def _(df):
    df.drop(df[df['topic_label'] == 'Criticism of Trump and Support for Democratic Policies'].index, inplace=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Investigate number of root posts under different topics on 2 platforms""")
    return


@app.cell
def _(df_2, plt, sns):
    topic_order = df_2.groupby('topic_label')['index'].nunique().sort_values(ascending=False).index
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    platforms = ['bsky', 'ts']
    for (idx, platform) in enumerate(platforms):
        platform_data = df_2[df_2['platform'] == platform]
        root_post_counts = platform_data.groupby('topic_label')['index'].nunique().reindex(topic_order)
        sns.barplot(x=root_post_counts.values, y=root_post_counts.index, ax=axes[idx], palette='tab20')
        axes[idx].set_title(f'Unique Root Posts per Topic on {platform.upper()}')
        axes[idx].set_xlabel('Number of Unique Root Posts')
        axes[idx].set_ylabel('Topic')
        for (j, (value, label)) in enumerate(zip(root_post_counts.values, root_post_counts.index)):
            axes[idx].annotate(f'{value}', xy=(value, j), xytext=(5, 0), textcoords='offset points', va='center', ha='left', fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1.a Investigate number of root posts under different partisanship on 2 platforms""")
    return


@app.cell
def _(df_2, plt, sns):
    (fig_1, axes_1) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    platforms_1 = ['bsky', 'ts']
    for (idx_1, platform_1) in enumerate(platforms_1):
        platform_data_1 = df_2[df_2['platform'] == platform_1]
        partisanship_counts = platform_data_1.groupby('partisanship')['index'].nunique().reindex(['left', 'center', 'right'])
        sns.barplot(x=partisanship_counts.values, y=partisanship_counts.index, ax=axes_1[idx_1], palette='tab20')
        axes_1[idx_1].set_title(f'Partisanship Distribution on {platform_1.upper()}')
        axes_1[idx_1].set_xlabel('Number of Root Posts')
        axes_1[idx_1].set_ylabel('Partisanship')
        for (j_1, (value_1, label_1)) in enumerate(zip(partisanship_counts.values, partisanship_counts.index)):
            axes_1[idx_1].annotate(f'{value_1}', xy=(value_1, j_1), xytext=(5, 0), textcoords='offset points', va='center', ha='left', fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Observation
        - The number of root posts under different topics on 2 platforms are different.
        - Presidential debates and Trump legal convictions are two topics that have the most root posts on both platforms.
        - For the rest of topics, it is clear that Truth Social is more aligned with conservative views and proporganda agenda, while Bluesky is more focus on recent policy and topcis are more diverse.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. general cross-platform comparison""")
    return


@app.cell
def _(ts_df_1):
    ts_df_1['outlier'].isna().sum()
    return


@app.cell
def _(df_2, ks_2samp, metrics, pd):
    ks_results = []
    for metric in metrics:
        bsky_vals = df_2[df_2['platform'] == 'bsky'][metric].dropna()
        ts_vals = df_2[df_2['platform'] == 'ts'][metric].dropna()
        ts_without_vals = df_2[df_2['outlier'] != True][metric].dropna()
        (ks_stat, p_value) = ks_2samp(bsky_vals, ts_vals)
        (ks_without_stat, p_without_value) = ks_2samp(bsky_vals, ts_without_vals)
        ks_results.append({'Metric': metric, 'KS Statistic': ks_stat, 'P-value': p_value})
        ks_results.append({'Metric': metric, 'KS Statistic': ks_without_stat, 'P-value': p_without_value})
    ks_overall_df = pd.DataFrame(ks_results)
    return


@app.cell
def _(np):
    # Function to compute CCDF
    def empirical_ccdf(data):
        sorted_data = np.sort(data)
        ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return sorted_data, ccdf
    return (empirical_ccdf,)


@app.cell
def _(df_2, empirical_ccdf, np, plt):
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted(df_2['topic_label'].unique()))))

    def compare_ccdf(data, metric, ax, by='partisanship', colors=None):
        """
        Plots CCDF for a given metric grouped by a categorical variable (e.g., partisanship).

        Parameters:
            data (DataFrame): The dataset
            metric (str): The numerical column to analyze
            ax (matplotlib.axis): Axis object to plot on
            by (str): Column to group by (default is "partisanship")
            colors (dict): Dictionary mapping categories to colors

        Returns:
            sorted_topics (list): Sorted list of group names
            partisan_groups (dict): Dictionary of metric values per group
        """
        sorted_topics = sorted(data[by].unique())
        partisan_groups = {}
        for topic in sorted_topics:
            subset = data[data[by] == topic][metric].dropna()
            (sorted_vals, ccdf_vals) = empirical_ccdf(subset)
            color = colors.get(topic, '#333333')
            ax.plot(sorted_vals, ccdf_vals, label=f'{topic.capitalize()}', linewidth=2, color=color)
            partisan_groups[topic] = subset.tolist()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(metric)
        ax.set_ylabel('CCDF (%)')
        ax.set_title(f'CCDF of {metric} by {by.capitalize()}')
        ax.grid()
        return (sorted_topics, partisan_groups)
    return (compare_ccdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure 1: CCDF of Max Depth, Size, Breadth, Structural Virality by Platform""")
    return


@app.cell
def _(df_2, metrics, np, plt):
    from matplotlib.ticker import LogLocator, NullFormatter
    plt.style.use('fivethirtyeight')
    colors_1 = {'bsky': '#5F9EA0', 'ts': '#F5BF03', 'outlier': '#FF6347'}
    (fig_2, axes_2) = plt.subplots(1, 4, figsize=(18, 5), dpi=300)
    for (idx_2, metric_1) in enumerate(metrics):
        for platform_2 in ['bsky', 'ts']:
            if platform_2 == 'outlier':
                values = df_2[df_2['outlier'] != True][metric_1].dropna()
            else:
                values = df_2[df_2['platform'] == platform_2][metric_1].dropna()
            sorted_vals = np.sort(values)
            ccdf = 1 - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            axes_2[idx_2].plot(sorted_vals, ccdf, label=platform_2.capitalize(), linewidth=2.5, color=colors_1[platform_2])
        metric_labels = {'max_depth': 'Cascade Depth', 'size': 'Cascade Size', 'breadth': 'Cascade Max-Breadth', 'structural_virality': 'Cascade Virality'}
        metric_str = metric_labels.get(metric_1, metric_1.capitalize())
        if metric_1 != 'structural_virality':
            axes_2[idx_2].set_xscale('log')
        axes_2[idx_2].set_yscale('log')
        axes_2[idx_2].set_xlabel(metric_str, fontsize=12)
        if idx_2 == 0:
            axes_2[idx_2].set_ylabel('CCDF', fontsize=12)
        axes_2[idx_2].tick_params(labelsize=10)
        axes_2[idx_2].xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        axes_2[idx_2].yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        axes_2[idx_2].yaxis.set_minor_formatter(NullFormatter())
        for spine in ['top', 'right']:
            axes_2[idx_2].spines[spine].set_visible(False)
        axes_2[idx_2].set_xlim(sorted_vals.min(), sorted_vals.max())
        axes_2[idx_2].set_ylim(0.0001, 1)
    plt.legend(labels=['BlueSky', 'TruthSocial', 'TruthSocial (No Influencers)'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_2, ks_2samp, metrics, np, pd, plt):
    df_std = df_2.copy()
    mask_bsky = df_std['platform'] == 'bsky'
    df_std.loc[mask_bsky, metrics] = (df_std.loc[mask_bsky, metrics] - df_std.loc[mask_bsky, metrics].mean()) / df_std.loc[mask_bsky, metrics].std()
    mask_ts = df_std['platform'] == 'ts'
    df_std.loc[mask_ts, metrics] = (df_std.loc[mask_ts, metrics] - df_std.loc[mask_ts, metrics].mean()) / df_std.loc[mask_ts, metrics].std()
    ks_results_1 = []
    for metric_2 in metrics:
        bsky_vals_1 = df_std[df_std['platform'] == 'bsky'][metric_2].dropna()
        ts_vals_1 = df_std[df_std['platform'] == 'ts'][metric_2].dropna()
        (ks_stat_1, p_value_1) = ks_2samp(bsky_vals_1, ts_vals_1)
        ks_results_1.append({'Metric': metric_2, 'KS Statistic': ks_stat_1, 'P-value': p_value_1})
    ks_overall_df_std = pd.DataFrame(ks_results_1)
    (fig_3, axes_3) = plt.subplots(1, 4, figsize=(18, 6))
    for (idx_3, metric_2) in enumerate(metrics):
        for platform_3 in ['bsky', 'ts']:
            values_1 = df_std[df_std['platform'] == platform_3][metric_2].dropna()
            sorted_vals_1 = np.sort(values_1)
            ccdf_std = 1 - np.arange(1, len(sorted_vals_1) + 1) / len(sorted_vals_1)
            axes_3[idx_3].plot(sorted_vals_1, ccdf_std, label=f'{platform_3}', linewidth=2)
        axes_3[idx_3].set_yscale('log')
        axes_3[idx_3].set_xlabel(metric_2)
        axes_3[idx_3].set_ylabel('CCdf_std (%)')
        axes_3[idx_3].set_title(f'CCdf_std of {metric_2} by Platform')
        axes_3[idx_3].grid()
        (ks_stat_1, p_value_1) = ks_overall_df_std.loc[ks_overall_df_std['Metric'] == metric_2, ['KS Statistic', 'P-value']].values[0]
        axes_3[idx_3].text(0.6, 0.1, f'KS={ks_stat_1:.4f}\nP={p_value_1:.4f}', transform=axes_3[idx_3].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. general cross-topic comparison""")
    return


@app.cell
def _(compare_ccdf, df_2, metrics, plt):
    (fig_4, axes_4) = plt.subplots(1, 4, figsize=(18, 6))
    axes_4 = axes_4.flatten()
    for (idx_4, metric_3) in enumerate(metrics):
        compare_ccdf(df_2, metric_3, axes_4[idx_4])
    (handles, labels) = axes_4[0].get_legend_handles_labels()
    fig_4.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. comparison by platform & by topic""")
    return


@app.cell
def _(compare_ccdf, df_2, metrics, plt, sorted_topics):
    for platform_4 in ['bsky', 'ts']:
        platform_data_2 = df_2[df_2['platform'] == platform_4]
        (fig_5, axes_5) = plt.subplots(1, 4, figsize=(18, 6))
        axes_5 = axes_5.flatten()
        for (idx_5, metric_4) in enumerate(metrics):
            compare_ccdf(platform_data_2, metric_4, axes_5[idx_5])
            axes_5[idx_5].set_title(f'{platform_4.upper()} - {metric_4}')
        (handles_1, labels_1) = zip(*sorted(zip(axes_5[0].get_legend_handles_labels()[0], axes_5[0].get_legend_handles_labels()[1]), key=lambda x: sorted_topics.index(x[1])))
        fig_5.legend(handles_1, labels_1, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(df_2):
    df_2['partisanship'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Comparison by partisanship""")
    return


@app.cell
def _(compare_ccdf, df_2, ks_2samp, metrics, plt):
    partisan_colors = {'left': '#82a6c0', 'center': '#bbcd78', 'right': '#e26a69'}
    for platform_5 in ['bsky', 'ts']:
        platform_data_3 = df_2[df_2['platform'] == platform_5]
        (fig_6, axes_6) = plt.subplots(1, 4, figsize=(18, 6), dpi=300)
        axes_6 = axes_6.flatten()
        ks_significant = {}
        for (idx_6, metric_5) in enumerate(metrics):
            (sorted_topics, partisan_groups) = compare_ccdf(platform_data_3, metric_5, axes_6[idx_6], by='partisanship', colors=partisan_colors)
            if metric_5 == 'max_depth':
                metric_str_1 = 'Depth'
            elif metric_5 == 'size':
                metric_str_1 = 'Size'
            elif metric_5 == 'breadth':
                metric_str_1 = 'Breadth'
            elif metric_5 == 'structural_virality':
                metric_str_1 = 'Structural Virality'
            axes_6[idx_6].set_title(f'{platform_5.upper()} - {metric_str_1}')
            ks_results_2 = {}
            partisan_keys = list(partisan_groups.keys())
            for i_1 in range(len(partisan_keys)):
                for j_2 in range(i_1 + 1, len(partisan_keys)):
                    (group1, group2) = (partisan_keys[i_1], partisan_keys[j_2])
                    (ks_stat_2, p_value_2) = ks_2samp(partisan_groups[group1], partisan_groups[group2])
                    ks_results_2[f'{group1} vs {group2}'] = p_value_2
                    if p_value_2 < 0.05:
                        ks_significant[group1] = True
                        ks_significant[group2] = True
        (legend_handles, legend_labels) = axes_6[0].get_legend_handles_labels()
        sorted_topics_lower = {topic.lower(): topic for topic in sorted_topics}
        sorted_legend = sorted(zip(legend_handles, legend_labels), key=lambda x: sorted_topics_lower.get(x[1].lower(), x[1]))
        (handles_2, labels_2) = zip(*sorted_legend)
        formatted_labels = [f'{label}' if ks_significant.get(label.lower(), False) else label for label in labels_2]
        fig_6.legend(handles_2, formatted_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        plt.tight_layout()
        plt.show()
    return (sorted_topics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# B. Reposts Network""")
    return


@app.cell
def _(pd):
    rp_bsky_df = pd.read_csv("../data/bsky_repost_stats.csv")
    rp_ts_df = pd.read_csv("../data/ts_repost_stats.csv")

    rp_ts_df.rename(columns={"topic": "topic_label"}, inplace=True)
    return rp_bsky_df, rp_ts_df


@app.cell
def _(rp_bsky_df):
    rp_bsky_df.reset_index(inplace=True)
    return


@app.cell
def _(bsky_post_to_label, pd, rp_bsky_df, rp_ts_df, ts_post_to_label):
    for (i_2, row_1) in rp_ts_df.iterrows():
        try:
            rp_ts_df.loc[i_2, 'partisanship'] = ts_post_to_label[row_1['post']]
        except:
            rp_ts_df.loc[i_2, 'partisanship'] = 'center'
    for (i_2, row_1) in rp_bsky_df.iterrows():
        try:
            rp_bsky_df.loc[i_2, 'partisanship'] = bsky_post_to_label[row_1['post']]
        except:
            rp_bsky_df.loc[i_2, 'partisanship'] = 'center'
    rp_ts_df['partisanship'] = rp_ts_df['partisanship'].replace({'error': 'center'})
    rp_bsky_df['partisanship'] = rp_bsky_df['partisanship'].replace({'error': 'center'})
    rp_df = pd.concat([rp_bsky_df, rp_ts_df], ignore_index=True)
    rp_df = rp_df.dropna(subset=['topic_label'])
    return


@app.cell
def _(rp_bsky_df):
    rp_bsky_df.sort_values(by='size', ascending=False)
    return


@app.cell
def _(columns_to_keep):
    columns_to_keep
    return


@app.cell
def _(columns_to_keep, pd, rp_bsky_df, rp_ts_df):
    rp_bsky_df_1 = rp_bsky_df[columns_to_keep].reset_index(drop=True)
    rp_ts_df_1 = rp_ts_df[columns_to_keep].reset_index(drop=True)
    rp_df_1 = pd.concat([rp_bsky_df_1, rp_ts_df_1], ignore_index=True)
    rp_df_1 = rp_df_1.dropna(subset=['topic_label'])
    return rp_bsky_df_1, rp_df_1, rp_ts_df_1


@app.cell
def _(rp_df_1):
    rp_df_1['topic_label'] = rp_df_1['topic_label'].replace({'MAGA and Pro-Trump Hashtags and Advocacy': 'Pro-Trump and MAGA Advocacy'})
    rp_df_2 = rp_df_1[rp_df_1['topic_label'] != 'Criticism of Trump and Support for Democratic Policies']
    return (rp_df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Investigate number of reposts under different topics on 2 platforms""")
    return


@app.cell
def _(plt, rp_df_2, sns):
    topic_order_1 = rp_df_2.groupby('topic_label')['index'].nunique().sort_values(ascending=False).index
    (fig_7, axes_7) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    platforms_2 = ['bsky', 'ts']
    for (idx_7, platform_6) in enumerate(platforms_2):
        platform_data_4 = rp_df_2[rp_df_2['platform'] == platform_6]
        root_post_counts_1 = platform_data_4.groupby('topic_label')['index'].nunique().reindex(topic_order_1)
        sns.barplot(x=root_post_counts_1.values, y=root_post_counts_1.index, ax=axes_7[idx_7], palette='tab20')
        axes_7[idx_7].set_title(f'Unique Root Posts per Topic on {platform_6.upper()}')
        axes_7[idx_7].set_xlabel('Number of Unique Root Posts')
        axes_7[idx_7].set_ylabel('Topic')
        for (j_3, (value_2, label_2)) in enumerate(zip(root_post_counts_1.values, root_post_counts_1.index)):
            axes_7[idx_7].annotate(f'{value_2}', xy=(value_2, j_3), xytext=(5, 0), textcoords='offset points', va='center', ha='left', fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.a Investigate number of reposts under different partisanship on 2 platforms""")
    return


@app.cell
def _(plt, rp_df_2, sns):
    (fig_8, axes_8) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    platforms_3 = ['bsky', 'ts']
    for (idx_8, platform_7) in enumerate(platforms_3):
        platform_data_5 = rp_df_2[rp_df_2['platform'] == platform_7]
        partisanship_counts_1 = platform_data_5.groupby('partisanship')['index'].nunique().reindex(['left', 'center', 'right'])
        sns.barplot(x=partisanship_counts_1.values, y=partisanship_counts_1.index, ax=axes_8[idx_8], palette='tab20')
        axes_8[idx_8].set_title(f'Partisanship Distribution on {platform_7.upper()}')
        axes_8[idx_8].set_xlabel('Number of Root Posts')
        axes_8[idx_8].set_ylabel('Partisanship')
        for (j_4, (value_3, label_3)) in enumerate(zip(partisanship_counts_1.values, partisanship_counts_1.index)):
            axes_8[idx_8].annotate(f'{value_3}', xy=(value_3, j_4), xytext=(5, 0), textcoords='offset points', va='center', ha='left', fontsize=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. General cross-platform comparison""")
    return


@app.cell
def _(ks_2samp, pd, rp_df_2):
    rp_metrics = ['max_depth', 'size', 'breadth', 'structural_virality']
    ks_results_3 = []
    for metric_6 in rp_metrics:
        bsky_vals_2 = rp_df_2[rp_df_2['platform'] == 'bsky'][metric_6].dropna()
        ts_vals_2 = rp_df_2[rp_df_2['platform'] == 'ts'][metric_6].dropna()
        (ks_stat_3, p_value_3) = ks_2samp(bsky_vals_2, ts_vals_2)
        ks_results_3.append({'Metric': metric_6, 'KS Statistic': ks_stat_3, 'P-value': p_value_3})
    ks_overall_df_1 = pd.DataFrame(ks_results_3)
    return ks_overall_df_1, rp_metrics


@app.cell
def _(ks_overall_df_1, np, plt, rp_df_2, rp_metrics):
    colors_2 = {'bsky': '#007FFF', 'ts': '#FFD700', 'ts_without_trump': '#8B0000'}
    (fig_9, axes_9) = plt.subplots(1, 4, figsize=(18, 6), dpi=300)
    for (idx_9, metric_7) in enumerate(rp_metrics):
        for platform_8 in ['bsky', 'ts']:
            values_2 = rp_df_2[rp_df_2['platform'] == platform_8][metric_7].dropna()
            sorted_vals_2 = np.sort(values_2)
            ccdf_1 = 1 - np.arange(1, len(sorted_vals_2) + 1) / len(sorted_vals_2)
            axes_9[idx_9].plot(sorted_vals_2, ccdf_1, label=f"{platform_8.replace('_', ' ').title()}", linewidth=2, color=colors_2[platform_8])
        metric_mapping = {'max_depth': 'Depth', 'size': 'Size', 'breadth': 'Breadth', 'structural_virality': 'Structural Virality'}
        metric_str_2 = metric_mapping.get(metric_7, metric_7)
        if metric_7 != 'structural_virality':
            axes_9[idx_9].set_xscale('log')
        axes_9[idx_9].set_yscale('log')
        axes_9[idx_9].set_xlabel(metric_str_2)
        axes_9[idx_9].set_ylabel('CCDF (%)')
        axes_9[idx_9].set_title(f'CCDF of {metric_str_2} by Platform')
        axes_9[idx_9].grid()
        (ks_stat_4, p_value_4) = ks_overall_df_1.loc[ks_overall_df_1['Metric'] == metric_7, ['KS Statistic', 'P-value']].values[0]
        axes_9[idx_9].text(0.6, 0.1, f'KS={ks_stat_4:.4f}\nP={p_value_4:.4f}', transform=axes_9[idx_9].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(labels=['BlueSky', 'TruthSocial'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# FIGURE 1:""")
    return


@app.cell
def _(df_2, np, plt, rp_df_2):
    repost_colors = {'bsky': '#5F9EA0', 'ts': '#FF6347', 'outlier': '#FF6347'}
    reply_colors = {'bsky': '#5F9EA0', 'ts': '#FF6347'}
    all_metrics = ['size', 'max_depth', 'breadth', 'structural_virality']
    metric_labels_1 = {'max_depth': 'Cascade Depth', 'size': 'Cascade Size', 'breadth': 'Cascade Max-Breadth', 'structural_virality': 'Cascade Virality'}
    (fig_10, axes_10) = plt.subplots(2, 4, figsize=(14, 7), dpi=300, constrained_layout=True)
    for (idx_10, metric_8) in enumerate(all_metrics):
        ax = axes_10[0, idx_10]
        for platform_9 in ['bsky', 'ts']:
            if platform_9 == 'outlier':
                values_3 = df_2[df_2['outlier'] != True][metric_8].dropna()
            else:
                values_3 = df_2[df_2['platform'] == platform_9][metric_8].dropna()
            sorted_vals_3 = np.sort(values_3)
            ccdf_2 = 1 - np.arange(1, len(sorted_vals_3) + 1) / len(sorted_vals_3)
            ax.plot(sorted_vals_3, ccdf_2, label=platform_9.capitalize(), linewidth=3, color=repost_colors[platform_9], alpha=0.9)
        if metric_8 != 'structural_virality':
            ax.set_xscale('log')
        ax.set_yscale('log')
        if idx_10 == 0:
            ax.set_ylabel('CCDF', fontsize=20, fontweight='bold')
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='x', labelbottom=True)
        ax.tick_params(labelsize=18)
        for spine_1 in ['top', 'right']:
            ax.spines[spine_1].set_visible(False)
    for (idx_10, metric_8) in enumerate(all_metrics):
        ax = axes_10[1, idx_10]
        for platform_9 in ['bsky', 'ts']:
            values_3 = rp_df_2[rp_df_2['platform'] == platform_9][metric_8].dropna()
            sorted_vals_3 = np.sort(values_3)
            ccdf_2 = 1 - np.arange(1, len(sorted_vals_3) + 1) / len(sorted_vals_3)
            ax.plot(sorted_vals_3, ccdf_2, label=platform_9.capitalize(), linewidth=3, color=reply_colors[platform_9], alpha=0.9)
        if metric_8 != 'structural_virality':
            ax.set_xscale('log')
        ax.set_yscale('log')
        if idx_10 == 0:
            ax.set_ylabel('CCDF', fontsize=20, fontweight='bold')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(metric_labels_1.get(metric_8, metric_8), fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=18)
        for spine_1 in ['top', 'right']:
            ax.spines[spine_1].set_visible(False)
    axes_10[0, 3].annotate('Repost', xy=(0.8, 0.8), xycoords='axes fraction', fontsize=15, fontweight='bold')
    axes_10[1, 3].annotate('Reply', xy=(0.8, 0.8), xycoords='axes fraction', fontsize=15, fontweight='bold')
    subplot_labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
    for (idx_10, ax) in enumerate(axes_10.flat):
        ax.text(0.02, 0.95, subplot_labels[idx_10], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')
    (handles_3, labels_3) = axes_10[0, 0].get_legend_handles_labels()
    fig_10.legend(handles_3, ['BlueSky', 'TruthSocial', 'TruthSocial (No Influencers)'], loc='upper center', bbox_to_anchor=(0.13, 0.68), ncol=1, frameon=False, fontsize=13)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. General cross-topic comparison""")
    return


@app.cell
def _(plt, rp_bsky_df_1, rp_ts_df_1):
    plt.figure(rows=2, cols=1)
    plt.scatter(rp_bsky_df_1['max_depth'], rp_bsky_df_1['size'])
    plt.scatter(rp_ts_df_1['max_depth'], rp_ts_df_1['size'])
    plt.show()
    return


@app.cell
def _(compare_ccdf, metrics, plt, rp_df_2):
    (fig_11, axes_11) = plt.subplots(1, 4, figsize=(18, 6))
    axes_11 = axes_11.flatten()
    for (idx_11, metric_9) in enumerate(metrics):
        compare_ccdf(rp_df_2, metric_9, axes_11[idx_11])
    (handles_4, labels_4) = axes_11[0].get_legend_handles_labels()
    fig_11.legend(handles_4, labels_4, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Comparison by platform & by topic""")
    return


@app.cell
def _(compare_ccdf, metrics, plt, rp_df_2, sorted_topics):
    for platform_10 in ['bsky', 'ts']:
        platform_data_6 = rp_df_2[rp_df_2['platform'] == platform_10]
        (fig_12, axes_12) = plt.subplots(1, 4, figsize=(18, 6))
        axes_12 = axes_12.flatten()
        for (idx_12, metric_10) in enumerate(metrics):
            compare_ccdf(platform_data_6, metric_10, axes_12[idx_12])
            axes_12[idx_12].set_title(f'{platform_10.upper()} - {metric_10}')
        (handles_5, labels_5) = zip(*sorted(zip(axes_12[0].get_legend_handles_labels()[0], axes_12[0].get_legend_handles_labels()[1]), key=lambda x: sorted_topics.index(x[1])))
        fig_12.legend(handles_5, labels_5, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Comparison by partisanship""")
    return


@app.cell
def _(compare_ccdf, ks_2samp, metrics, plt, rp_df_2):
    partisan_colors_1 = {'left': '#82a6c0', 'center': '#bbcd78', 'right': '#e26a69'}
    for platform_11 in ['bsky', 'ts']:
        platform_data_7 = rp_df_2[rp_df_2['platform'] == platform_11]
        (fig_13, axes_13) = plt.subplots(1, 4, figsize=(18, 6), dpi=300)
        axes_13 = axes_13.flatten()
        ks_significant_1 = {}
        for (idx_13, metric_11) in enumerate(metrics):
            (sorted_topics_1, partisan_groups_1) = compare_ccdf(platform_data_7, metric_11, axes_13[idx_13], by='partisanship', colors=partisan_colors_1)
            if metric_11 == 'max_depth':
                metric_str_3 = 'Depth'
            elif metric_11 == 'size':
                metric_str_3 = 'Size'
            elif metric_11 == 'breadth':
                metric_str_3 = 'Breadth'
            elif metric_11 == 'structural_virality':
                metric_str_3 = 'Structural Virality'
            axes_13[idx_13].set_title(f'{platform_11.upper()} - {metric_str_3}')
            ks_results_4 = {}
            partisan_keys_1 = list(partisan_groups_1.keys())
            for i_3 in range(len(partisan_keys_1)):
                for j_5 in range(i_3 + 1, len(partisan_keys_1)):
                    (group1_1, group2_1) = (partisan_keys_1[i_3], partisan_keys_1[j_5])
                    (ks_stat_5, p_value_5) = ks_2samp(partisan_groups_1[group1_1], partisan_groups_1[group2_1])
                    ks_results_4[f'{group1_1} vs {group2_1}'] = p_value_5
                    if p_value_5 < 0.05:
                        ks_significant_1[group1_1] = True
                        ks_significant_1[group2_1] = True
        (legend_handles_1, legend_labels_1) = axes_13[0].get_legend_handles_labels()
        sorted_topics_lower_1 = {topic.lower(): topic for topic in sorted_topics_1}
        sorted_legend_1 = sorted(zip(legend_handles_1, legend_labels_1), key=lambda x: sorted_topics_lower_1.get(x[1].lower(), x[1]))
        (handles_6, labels_6) = zip(*sorted_legend_1)
        formatted_labels_1 = [f'{label}' if ks_significant_1.get(label.lower(), False) else label for label in labels_6]
        fig_13.legend(handles_6, formatted_labels_1, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        plt.tight_layout()
        plt.show()
    return (sorted_topics_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# C. Combined Network""")
    return


@app.cell
def _(pd):
    c_bsky_df = pd.read_csv("../data/bsky_combined_stats.csv")
    c_ts_df = pd.read_csv("../data/ts_combined_stats.csv")

    c_ts_df.rename(columns={"topic": "topic_label"}, inplace=True)
    return c_bsky_df, c_ts_df


@app.cell
def _(c_bsky_df, c_ts_df, columns_to_keep, pd):
    c_bsky_df_1 = c_bsky_df[columns_to_keep].reset_index(drop=True)
    c_ts_df_1 = c_ts_df[columns_to_keep].reset_index(drop=True)
    c_df = pd.concat([c_bsky_df_1, c_ts_df_1], ignore_index=True)
    c_df = c_df.dropna(subset=['topic_label'])
    return (c_df,)


@app.cell
def _(c_df):
    # Merge similar topics
    c_df["topic_label"] = c_df["topic_label"].replace(
        {"MAGA and Pro-Trump Hashtags and Advocacy": "Pro-Trump and MAGA Advocacy"}
    )
    return


@app.cell
def _(c_df):
    c_df_1 = c_df[c_df['topic_label'] != 'Criticism of Trump and Support for Democratic Policies']
    return (c_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Investigate number of root posts under different topics on 2 platforms""")
    return


@app.cell
def _(c_df_1, df_2, plt, sns):
    topic_order_2 = c_df_1.groupby('topic_label')['index'].nunique().sort_values(ascending=False).index
    (fig_14, axes_14) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    platforms_4 = ['bsky', 'ts']
    for (idx_14, platform_12) in enumerate(platforms_4):
        platform_data_8 = df_2[df_2['platform'] == platform_12]
        root_post_counts_2 = platform_data_8.groupby('topic_label')['index'].nunique().reindex(topic_order_2)
        sns.barplot(x=root_post_counts_2.values, y=root_post_counts_2.index, ax=axes_14[idx_14], palette='tab20')
        axes_14[idx_14].set_title(f'Unique Root Posts per Topic on {platform_12.upper()}')
        axes_14[idx_14].set_xlabel('Number of Unique Root Posts')
        axes_14[idx_14].set_ylabel('Topic')
        for (j_6, (value_4, label_4)) in enumerate(zip(root_post_counts_2.values, root_post_counts_2.index)):
            axes_14[idx_14].annotate(f'{value_4}', xy=(value_4, j_6), xytext=(5, 0), textcoords='offset points', va='center', ha='left', fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. general cross-platform comparison""")
    return


@app.cell
def _(c_df_1, ks_2samp, metrics, pd):
    c_ks_results = []
    for metric_12 in metrics:
        c_bsky_vals = c_df_1[c_df_1['platform'] == 'bsky'][metric_12].dropna()
        c_ts_vals = c_df_1[c_df_1['platform'] == 'ts'][metric_12].dropna()
        (c_ks_stat, c_p_value) = ks_2samp(c_bsky_vals, c_ts_vals)
        c_ks_results.append({'Metric': metric_12, 'KS Statistic': c_ks_stat, 'P-value': c_p_value})
    c_ks_overall_df = pd.DataFrame(c_ks_results)
    return (c_ks_overall_df,)


@app.cell
def _(c_df_1, c_ks_overall_df, metrics, np, plt):
    (fig_15, axes_15) = plt.subplots(1, 4, figsize=(18, 6))
    for (idx_15, metric_13) in enumerate(metrics):
        for platform_13 in ['bsky', 'ts']:
            values_4 = c_df_1[c_df_1['platform'] == platform_13][metric_13].dropna()
            sorted_vals_4 = np.sort(values_4)
            ccdf_3 = 1 - np.arange(1, len(sorted_vals_4) + 1) / len(sorted_vals_4)
            axes_15[idx_15].plot(sorted_vals_4, ccdf_3, label=f'{platform_13}', linewidth=2)
        axes_15[idx_15].set_xscale('log')
        axes_15[idx_15].set_yscale('log')
        axes_15[idx_15].set_xlabel(metric_13)
        axes_15[idx_15].set_ylabel('CCDF (%)')
        axes_15[idx_15].set_title(f'CCDF of {metric_13} by Platform')
        axes_15[idx_15].grid()
        (c_ks_stat_1, p_value_6) = c_ks_overall_df.loc[c_ks_overall_df['Metric'] == metric_13, ['KS Statistic', 'P-value']].values[0]
        axes_15[idx_15].text(0.6, 0.1, f'KS={c_ks_stat_1:.4f}\nP={p_value_6:.4f}', transform=axes_15[idx_15].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. general cross-topic comparison""")
    return


@app.cell
def _(c_df_1, compare_ccdf, metrics, plt):
    (fig_16, axes_16) = plt.subplots(1, 4, figsize=(18, 6))
    axes_16 = axes_16.flatten()
    for (idx_16, metric_14) in enumerate(metrics):
        compare_ccdf(c_df_1, metric_14, axes_16[idx_16])
    (handles_7, labels_7) = axes_16[0].get_legend_handles_labels()
    fig_16.legend(handles_7, labels_7, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. comparison by platform & by topic""")
    return


@app.cell
def _(c_df_1, compare_ccdf, metrics, plt, sorted_topics_1):
    for platform_14 in ['bsky', 'ts']:
        platform_data_9 = c_df_1[c_df_1['platform'] == platform_14]
        (fig_17, axes_17) = plt.subplots(1, 4, figsize=(18, 6))
        axes_17 = axes_17.flatten()
        for (idx_17, metric_15) in enumerate(metrics):
            compare_ccdf(platform_data_9, metric_15, axes_17[idx_17])
            axes_17[idx_17].set_title(f'{platform_14.upper()} - {metric_15}')
        (handles_8, labels_8) = zip(*sorted(zip(axes_17[0].get_legend_handles_labels()[0], axes_17[0].get_legend_handles_labels()[1]), key=lambda x: sorted_topics_1.index(x[1])))
        fig_17.legend(handles_8, labels_8, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        plt.tight_layout()
        plt.show()
    return handles_8, labels_8


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Summary""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        1. Topic-wise comparison: Truth Social is more aligned with conservative views and proporganda agenda, while Bluesky is more focus on recent policy and topcis are more diverse.
        2. Topic-wise ccdf: 
            1. Reply network: most salient distribution for BlueSky is from presidential debates and same for Truth Social.
            2. Repost network: most salient distribution for BlueSky is from Trump's legal convictions and For Truth Social, it is from presidential debates.
            3. Combined network: as same as repost network, but the Truth Social phenomenon is more pronounced.
        3. 191575 out of 747571 for ts and 23138 out of 59813
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# D. Matching the cascades across platforms""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Understand the relationship between the size and other metrics""")
    return


@app.cell
def _(plt, rp_bsky_df_1, rp_ts_df_1):
    (fig_18, axes_18) = plt.subplots(1, 2, figsize=(14, 6))
    axes_18[0].scatter(rp_bsky_df_1['max_depth'], rp_bsky_df_1['log_size'], label='bsky', alpha=0.1)
    axes_18[1].scatter(rp_ts_df_1['max_depth'], rp_ts_df_1['log_size'], label='ts', alpha=0.1)
    axes_18[0].set_xlabel('max_depth')
    axes_18[0].set_ylabel('log_size')
    axes_18[0]
    axes_18[1].set_xlabel('max_depth')
    axes_18[1].set_ylabel('log_size')
    axes_18[0].set_title('bsky')
    axes_18[1].set_title('ts')
    plt.show()
    return


@app.cell
def _(plt, rp_bsky_df_1, rp_ts_df_1):
    (fig_19, axes_19) = plt.subplots(1, 2, figsize=(14, 6))
    axes_19[0].scatter(rp_bsky_df_1['breadth'], rp_bsky_df_1['log_size'], label='bsky', alpha=0.1)
    axes_19[1].scatter(rp_ts_df_1['breadth'], rp_ts_df_1['log_size'], label='ts', alpha=0.1)
    axes_19[0].set_xlabel('breadth')
    axes_19[0].set_ylabel('log_size')
    axes_19[1].set_xlabel('breadth')
    axes_19[1].set_ylabel('log_size')
    axes_19[0].set_title('bsky')
    axes_19[1].set_title('ts')
    plt.show()
    return


@app.cell
def _(ts_df_1):
    len(ts_df_1)
    return


@app.cell
def _(bsky_df_1, np, pd, ts_df_1):
    import random
    from collections import defaultdict
    ts_df_2 = ts_df_1.sort_values('size').reset_index()
    bsky_df_2 = bsky_df_1.sort_values('size').reset_index(drop=True)
    bsky_sizes = bsky_df_2['size'].to_numpy()
    ts_sizes = ts_df_2['size'].to_numpy()
    ts_indices = ts_df_2.index.to_numpy()
    matched_indices = []
    unmatched_size = set()
    size_to_indices = defaultdict(list)
    for (i_4, size) in enumerate(ts_sizes):
        size_to_indices[size].append(ts_indices[i_4])
    for size in bsky_sizes:
        idx_18 = np.searchsorted(ts_sizes, size)
        possible_matches = size_to_indices.get(ts_sizes[idx_18], [])
        if possible_matches:
            matched_indices.append(random.choice(possible_matches))
        else:
            unmatched_size.add(size)
    sampled_ts_df = ts_df_2.loc[matched_indices].reset_index(drop=True) if matched_indices else pd.DataFrame()
    return bsky_df_2, defaultdict, random, sampled_ts_df, ts_df_2


@app.cell
def _(sampled_ts_df):
    sampled_ts_df.shape
    return


@app.cell
def _(bsky_df_2, pd, ts_df_2):
    columns_to_keep_1 = ['platform', 'topic_label', 'max_depth', 'size', 'breadth', 'index', 'structural_virality']
    bsky_df_3 = bsky_df_2[columns_to_keep_1].reset_index(drop=True)
    ts_df_3 = ts_df_2[columns_to_keep_1 + 'outlier'].reset_index(drop=True)
    df_3 = pd.concat([bsky_df_3, ts_df_3], ignore_index=True)
    df_3 = df_3.dropna(subset=['topic_label'])
    metrics_1 = ['max_depth', 'size', 'breadth', 'structural_virality']
    return bsky_df_3, df_3, metrics_1, ts_df_3


@app.cell
def _(df_3, ks_2samp, metrics_1, pd):
    ks_results_5 = []
    for metric_16 in metrics_1:
        bsky_vals_3 = df_3[df_3['platform'] == 'bsky'][metric_16].dropna()
        ts_vals_3 = df_3[df_3['platform'] == 'ts'][metric_16].dropna()
        (ks_stat_6, p_value_7) = ks_2samp(bsky_vals_3, ts_vals_3)
        ks_results_5.append({'Metric': metric_16, 'KS Statistic': ks_stat_6, 'P-value': p_value_7})
    ks_overall_df_2 = pd.DataFrame(ks_results_5)
    return (ks_overall_df_2,)


@app.cell
def _(df_3, empirical_ccdf, np, plt):
    colors_3 = plt.cm.tab20(np.linspace(0, 1, len(sorted(df_3['topic_label'].unique()))))
    sorted_topics_2 = sorted(df_3['topic_label'].unique())
    topic_color_map = {topic: colors_3[i] for (i, topic) in enumerate(sorted_topics_2)}

    def compare_ccdf_1(data, metric, ax):
        for topic in sorted_topics_2:
            subset = data[data['topic_label'] == topic][metric].dropna()
            (sorted_vals, ccdf_vals) = empirical_ccdf(subset)
            ax.plot(sorted_vals, ccdf_vals, label=f'{topic}', linewidth=2, color=topic_color_map[topic])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(metric)
        ax.set_ylabel('CCDF (%)')
        ax.set_title(f'CCDF of {metric} by Topic')
        ax.grid()
    return


@app.cell
def _(df_3, ks_overall_df_2, metrics_1, np, plt):
    (fig_20, axes_20) = plt.subplots(1, 4, figsize=(18, 6))
    for (idx_19, metric_17) in enumerate(metrics_1):
        for platform_15 in ['bsky', 'ts']:
            values_5 = df_3[df_3['platform'] == platform_15][metric_17].dropna()
            sorted_vals_5 = np.sort(values_5)
            ccdf_4 = 1 - np.arange(1, len(sorted_vals_5) + 1) / len(sorted_vals_5)
            axes_20[idx_19].plot(sorted_vals_5, ccdf_4, label=f'{platform_15}', linewidth=2)
        axes_20[idx_19].set_yscale('log')
        axes_20[idx_19].set_xlabel(metric_17)
        axes_20[idx_19].set_ylabel('CCDF (%)')
        axes_20[idx_19].set_title(f'CCDF of {metric_17} by Platform')
        axes_20[idx_19].grid()
        (ks_stat_7, p_value_8) = ks_overall_df_2.loc[ks_overall_df_2['Metric'] == metric_17, ['KS Statistic', 'P-value']].values[0]
        axes_20[idx_19].text(0.6, 0.1, f'KS={ks_stat_7:.4f}\nP={p_value_8:.4f}', transform=axes_20[idx_19].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(bsky_df_3, defaultdict, np, plt, random, ts_df_3):
    from tqdm import tqdm
    columns_to_keep_2 = ['platform', 'topic_label', 'max_depth', 'size', 'breadth', 'index', 'structural_virality']
    bsky_df_4 = bsky_df_3[columns_to_keep_2].reset_index(drop=True)

    def compute_ccdf(data):
        sorted_data = np.sort(data)
        ccdf = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return (sorted_data, ccdf)
    ts_df_4 = ts_df_3.sort_values('size').reset_index(drop=True)
    bsky_df_4 = bsky_df_4.sort_values('size').reset_index(drop=True)
    bsky_sizes_1 = bsky_df_4['size'].to_numpy()
    ts_sizes_1 = ts_df_4['size'].to_numpy()
    ts_indices_1 = ts_df_4.index.to_numpy()
    size_to_indices_1 = defaultdict(list)
    for (i_5, size_1) in enumerate(ts_sizes_1):
        size_to_indices_1[size_1].append(ts_indices_1[i_5])
    num_bootstrap_samples = 1000
    bootstrap_ccdfs = {metric: [] for metric in ['max_depth', 'size', 'breadth', 'structural_virality']}
    for _ in tqdm(range(num_bootstrap_samples), desc='Bootstrapping samples'):
        matched_indices_1 = []
        for size_1 in bsky_sizes_1:
            idx_20 = np.searchsorted(ts_sizes_1, size_1)
            possible_matches_1 = size_to_indices_1.get(ts_sizes_1[idx_20], [])
            if possible_matches_1:
                matched_indices_1.append(random.choice(possible_matches_1))
        if matched_indices_1:
            sampled_ts_df_1 = ts_df_4.loc[matched_indices_1].reset_index(drop=True)
            sampled_ts_df_1 = sampled_ts_df_1[columns_to_keep_2]
            for metric_18 in bootstrap_ccdfs.keys():
                sample_vals = sampled_ts_df_1[metric_18].dropna().to_numpy()
                (sorted_vals_6, ccdf_vals) = compute_ccdf(sample_vals)
                bootstrap_ccdfs[metric_18].append((sorted_vals_6, ccdf_vals))
    ccdf_summary = {}
    for (metric_18, samples) in tqdm(bootstrap_ccdfs.items(), desc='Processing CCDFs'):
        all_x_vals = [x_vals for (x_vals, _) in samples]
        min_x = min((min(x) for x in all_x_vals if len(x) > 0))
        max_x = max((max(x) for x in all_x_vals if len(x) > 0))
        common_x_vals = np.linspace(min_x, max_x, 1000)
        interpolated_ccdfs = [np.interp(common_x_vals, x_vals, ccdf, left=1.0, right=0.0) for (x_vals, ccdf) in samples]
        interpolated_ccdfs = np.array(interpolated_ccdfs)
        ccdf_summary[metric_18] = {'mean': np.mean(interpolated_ccdfs, axis=0), 'lower': np.percentile(interpolated_ccdfs, 2.5, axis=0), 'upper': np.percentile(interpolated_ccdfs, 97.5, axis=0), 'x_vals': common_x_vals}
    (fig_21, axes_21) = plt.subplots(1, 4, figsize=(18, 6))
    for (idx_20, metric_18) in enumerate(ccdf_summary.keys()):
        bsky_vals_4 = bsky_df_4[metric_18].dropna()
        (sorted_vals_6, ccdf_5) = compute_ccdf(bsky_vals_4)
        axes_21[idx_20].plot(sorted_vals_6, ccdf_5, label='bsky', linewidth=2, color='b')
        summary = ccdf_summary[metric_18]
        axes_21[idx_20].plot(summary['x_vals'], summary['mean'], label='ts (bootstrap)', color='r')
        axes_21[idx_20].fill_between(summary['x_vals'], summary['lower'], summary['upper'], color='r', alpha=0.3)
        axes_21[idx_20].set_yscale('log')
        axes_21[idx_20].set_xlabel(metric_18)
        axes_21[idx_20].set_ylabel('CCDF (%)')
        axes_21[idx_20].set_title(f'CCDF of {metric_18} by Platform')
        axes_21[idx_20].grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.show()
    return bsky_df_4, compute_ccdf, sampled_ts_df_1, tqdm, ts_df_4


@app.cell
def _(bsky_df_4, plt, sampled_ts_df_1):
    (fig_22, axes_22) = plt.subplots(1, 2, figsize=(14, 6))
    axes_22[0].scatter(bsky_df_4['max_depth'], bsky_df_4['breadth'], label='bsky', alpha=0.2)
    axes_22[1].scatter(sampled_ts_df_1['max_depth'], sampled_ts_df_1['breadth'], label='ts', alpha=0.2)
    axes_22[0].set_xlabel('max_depth')
    axes_22[0].set_ylabel('breadth')
    axes_22[1].set_xlabel('max_depth')
    axes_22[1].set_ylabel('breadth')
    axes_22[0].set_title('bsky')
    axes_22[1].set_title('ts')
    plt.show()
    return


@app.cell
def _(defaultdict, np, pd, random, rp_bsky_df_1, rp_ts_df_1):
    rp_ts_df_2 = rp_ts_df_1.sort_values('size').reset_index(drop=True)
    rp_bsky_df_2 = rp_bsky_df_1.sort_values('size').reset_index(drop=True)
    bsky_sizes_2 = rp_bsky_df_2['size'].to_numpy()
    ts_sizes_2 = rp_ts_df_2['size'].to_numpy()
    ts_indices_2 = rp_ts_df_2.index.to_numpy()
    matched_indices_2 = []
    unmatched_size_1 = set()
    size_to_indices_2 = defaultdict(list)
    for (i_6, size_2) in enumerate(ts_sizes_2):
        size_to_indices_2[size_2].append(ts_indices_2[i_6])
    for size_2 in bsky_sizes_2:
        idx_21 = np.searchsorted(ts_sizes_2, size_2)
        possible_matches_2 = size_to_indices_2.get(ts_sizes_2[idx_21], [])
        if possible_matches_2:
            matched_indices_2.append(random.choice(possible_matches_2))
        else:
            unmatched_size_1.add(size_2)
    sampled_rp_ts_df = rp_ts_df_2.loc[matched_indices_2].reset_index(drop=True) if matched_indices_2 else pd.DataFrame()
    return rp_bsky_df_2, rp_ts_df_2, sampled_rp_ts_df


@app.cell
def _(sampled_rp_ts_df):
    sampled_rp_ts_df.drop_duplicates(inplace=True)
    return


@app.cell
def _(sampled_rp_ts_df):
    sampled_rp_ts_df.shape
    return


@app.cell
def _(pd, rp_bsky_df_2, sampled_rp_ts_df):
    columns_to_keep_3 = ['platform', 'topic_label', 'max_depth', 'size', 'breadth', 'structural_virality']
    rp_bsky_df_3 = rp_bsky_df_2[columns_to_keep_3].reset_index(drop=True)
    sampled_rp_ts_df_1 = sampled_rp_ts_df[columns_to_keep_3].reset_index(drop=True)
    rp_df_3 = pd.concat([rp_bsky_df_3, sampled_rp_ts_df_1], ignore_index=True)
    rp_df_3 = rp_df_3.dropna(subset=['topic_label'])
    print(len(rp_df_3))
    return rp_bsky_df_3, rp_df_3


@app.cell
def _(ks_2samp, pd, rp_df_3):
    ks_results_6 = []
    metrics_2 = ['max_depth', 'size', 'breadth', 'structural_virality']
    for metric_19 in metrics_2:
        bsky_vals_5 = rp_df_3[rp_df_3['platform'] == 'bsky'][metric_19].dropna()
        ts_vals_4 = rp_df_3[rp_df_3['platform'] == 'ts'][metric_19].dropna()
        (ks_stat_8, p_value_9) = ks_2samp(bsky_vals_5, ts_vals_4)
        ks_results_6.append({'Metric': metric_19, 'KS Statistic': ks_stat_8, 'P-value': p_value_9})
    ks_overall_df_3 = pd.DataFrame(ks_results_6)
    return ks_overall_df_3, metrics_2


@app.cell
def _(ks_overall_df_3, metrics_2, np, plt, rp_df_3):
    (fig_23, axes_23) = plt.subplots(1, 4, figsize=(18, 6))
    for (idx_22, metric_20) in enumerate(metrics_2):
        for platform_16 in ['bsky', 'ts']:
            values_6 = rp_df_3[rp_df_3['platform'] == platform_16][metric_20].dropna()
            sorted_vals_7 = np.sort(values_6)
            ccdf_6 = 1 - np.arange(1, len(sorted_vals_7) + 1) / len(sorted_vals_7)
            axes_23[idx_22].plot(sorted_vals_7, ccdf_6, label=f'{platform_16}', linewidth=2)
        axes_23[idx_22].set_yscale('log')
        axes_23[idx_22].set_xlabel(metric_20)
        axes_23[idx_22].set_ylabel('CCDF (%)')
        axes_23[idx_22].set_title(f'CCDF of {metric_20} by Platform')
        axes_23[idx_22].grid()
        (ks_stat_9, p_value_10) = ks_overall_df_3.loc[ks_overall_df_3['Metric'] == metric_20, ['KS Statistic', 'P-value']].values[0]
        axes_23[idx_22].text(0.6, 0.1, f'KS={ks_stat_9:.4f}\nP={p_value_10:.4f}', transform=axes_23[idx_22].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    compute_ccdf,
    defaultdict,
    np,
    plt,
    random,
    rp_bsky_df_3,
    rp_ts_df_2,
    tqdm,
):
    columns_to_keep_4 = ['platform', 'topic_label', 'max_depth', 'size', 'breadth', 'index', 'structural_virality']
    rp_bsky_df_4 = rp_bsky_df_3[columns_to_keep_4].reset_index(drop=True)
    rp_ts_df_3 = rp_ts_df_2.sort_values('size').reset_index(drop=True)
    rp_bsky_df_4 = rp_bsky_df_4.sort_values('size').reset_index(drop=True)
    bsky_sizes_3 = rp_bsky_df_4['size'].to_numpy()
    ts_sizes_3 = rp_ts_df_3['size'].to_numpy()
    ts_indices_3 = rp_ts_df_3.index.to_numpy()
    size_to_indices_3 = defaultdict(list)
    for (i_7, size_3) in enumerate(ts_sizes_3):
        size_to_indices_3[size_3].append(ts_indices_3[i_7])
    num_bootstrap_samples_1 = 100
    bootstrap_ccdfs_1 = {metric: [] for metric in ['max_depth', 'size', 'breadth', 'structural_virality']}
    for _ in tqdm(range(num_bootstrap_samples_1), desc='Bootstrapping samples'):
        matched_indices_3 = []
        for size_3 in bsky_sizes_3:
            idx_23 = np.searchsorted(ts_sizes_3, size_3)
            possible_matches_3 = size_to_indices_3.get(ts_sizes_3[idx_23], [])
            if possible_matches_3:
                matched_indices_3.append(random.choice(possible_matches_3))
        if matched_indices_3:
            sampled_rp_ts_df_2 = rp_ts_df_3.loc[matched_indices_3].reset_index(drop=True)
            sampled_rp_ts_df_2 = sampled_rp_ts_df_2[columns_to_keep_4]
            for metric_21 in bootstrap_ccdfs_1.keys():
                sample_vals_1 = sampled_rp_ts_df_2[metric_21].dropna().to_numpy()
                (sorted_vals_8, ccdf_vals_1) = compute_ccdf(sample_vals_1)
                bootstrap_ccdfs_1[metric_21].append((sorted_vals_8, ccdf_vals_1))
    ccdf_summary_1 = {}
    for (metric_21, samples_1) in tqdm(bootstrap_ccdfs_1.items(), desc='Processing CCDFs'):
        all_x_vals_1 = [x_vals for (x_vals, _) in samples_1]
        min_x_1 = min((min(x) for x in all_x_vals_1 if len(x) > 0))
        max_x_1 = max((max(x) for x in all_x_vals_1 if len(x) > 0))
        common_x_vals_1 = np.linspace(min_x_1, max_x_1, 1000)
        interpolated_ccdfs_1 = [np.interp(common_x_vals_1, x_vals, ccdf, left=1.0, right=0.0) for (x_vals, ccdf) in samples_1]
        interpolated_ccdfs_1 = np.array(interpolated_ccdfs_1)
        ccdf_summary_1[metric_21] = {'mean': np.mean(interpolated_ccdfs_1, axis=0), 'lower': np.percentile(interpolated_ccdfs_1, 2.5, axis=0), 'upper': np.percentile(interpolated_ccdfs_1, 97.5, axis=0), 'x_vals': common_x_vals_1}
    (fig_24, axes_24) = plt.subplots(1, 4, figsize=(18, 6))
    for (idx_23, metric_21) in enumerate(ccdf_summary_1.keys()):
        bsky_vals_6 = rp_bsky_df_4[metric_21].dropna()
        (sorted_vals_8, ccdf_7) = compute_ccdf(bsky_vals_6)
        axes_24[idx_23].plot(sorted_vals_8, ccdf_7, label='bsky', linewidth=2, color='b')
        summary_1 = ccdf_summary_1[metric_21]
        axes_24[idx_23].plot(summary_1['x_vals'], summary_1['mean'], label='ts (bootstrap)', color='r')
        axes_24[idx_23].fill_between(summary_1['x_vals'], summary_1['lower'], summary_1['upper'], color='r', alpha=0.3)
        axes_24[idx_23].set_yscale('log')
        axes_24[idx_23].set_xlabel(metric_21)
        axes_24[idx_23].set_ylabel('CCDF (%)')
        axes_24[idx_23].set_title(f'CCDF of {metric_21} by Platform')
        axes_24[idx_23].grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.show()
    return rp_bsky_df_4, rp_ts_df_3


@app.cell
def _(df_3, plt):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    df_bsky = df_3[df_3['platform'] == 'bsky']
    df_ts = df_3[df_3['platform'] == 'ts']
    bsky_grouped = df_bsky.groupby('max_depth')['breadth'].mean().reset_index()
    ts_grouped = df_ts.groupby('max_depth')['breadth'].mean().reset_index()
    frac = 0.2
    bsky_smooth = lowess(bsky_grouped['breadth'], bsky_grouped['max_depth'], frac=frac)
    ts_smooth = lowess(ts_grouped['breadth'], ts_grouped['max_depth'], frac=frac)
    (fig_25, axes_25) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes_25[0].plot(bsky_smooth[:, 0], bsky_smooth[:, 1], linestyle='-', label='bsky')
    axes_25[1].plot(ts_smooth[:, 0], ts_smooth[:, 1], linestyle='-', label='ts')
    axes_25[0].set_title('bsky (LOESS Smoothed)')
    axes_25[1].set_title('ts (LOESS Smoothed)')
    for ax_1 in axes_25:
        ax_1.set_xlabel('max_depth')
        ax_1.set_ylabel('breadth')
        ax_1.legend()
    plt.show()
    return (lowess,)


@app.cell
def _(bsky_df_4, df_3, lowess, np, plt, sns, ts_df_4):
    df_3['partisanship'] = df_3['partisanship'].replace({'lean left': 'left', 'lean right': 'right'})
    ts_df_4['partisanship'] = ts_df_4['partisanship'].replace({'lean left': 'left', 'lean right': 'right'})
    bsky_df_4['partisanship'] = bsky_df_4['partisanship'].replace({'lean left': 'left', 'lean right': 'right'})
    bsky_grouped_1 = bsky_df_4.groupby(['partisanship', 'size'])['breadth'].mean().reset_index()
    ts_grouped_1 = ts_df_4.groupby(['partisanship', 'size'])['breadth'].mean().reset_index()
    unique_partisanship = list(set(ts_df_4['partisanship'].unique()) | set(bsky_df_4['partisanship'].unique()))
    shared_palette = sns.color_palette('tab10', len(unique_partisanship))
    palette_dict = dict(zip(unique_partisanship, shared_palette))
    (fig_26, axes_26) = plt.subplots(1, 2, figsize=(14, 6))
    ts_df_4['log_size'] = np.log10(ts_df_4['size'])
    bsky_df_4['log_size'] = np.log10(bsky_df_4['size'])
    ts_df_4['log_breadth'] = np.log10(ts_df_4['breadth'])
    bsky_df_4['log_breadth'] = np.log10(bsky_df_4['breadth'])
    frac_1 = 0.2
    for partisanship in unique_partisanship:
        partisanship_ts = ts_df_4[ts_df_4['partisanship'] == partisanship]
        partisanship_bsky = bsky_df_4[bsky_df_4['partisanship'] == partisanship]
        partisanship_ts_grouped = partisanship_ts.groupby('log_size')['log_breadth'].mean().reset_index()
        partisanship_bsky_grouped = partisanship_bsky.groupby('log_size')['log_breadth'].mean().reset_index()
        ts_loess = lowess(partisanship_ts_grouped['log_breadth'], partisanship_ts_grouped['log_size'], frac=frac_1)
        bsky_loess = lowess(partisanship_bsky_grouped['log_breadth'], partisanship_bsky_grouped['log_size'], frac=frac_1)
        axes_26[0].plot(ts_loess[:, 0], ts_loess[:, 1], color=palette_dict[partisanship], linewidth=2, label=partisanship)
        axes_26[1].plot(bsky_loess[:, 0], bsky_loess[:, 1], color=palette_dict[partisanship], linewidth=2, label=partisanship)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# E. Normalization""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Normalize the depth and width of the cascades by its size""")
    return


@app.cell
def _(bsky_df_4, rp_bsky_df_4, rp_ts_df_3, ts_df_4):
    bsky_df_4['bredth_normalized'] = bsky_df_4['breadth'] / bsky_df_4['size']
    ts_df_4['bredth_normalized'] = ts_df_4['breadth'] / ts_df_4['size']
    rp_bsky_df_4['bredth_normalized'] = rp_bsky_df_4['breadth'] / rp_bsky_df_4['size']
    rp_ts_df_3['bredth_normalized'] = rp_ts_df_3['breadth'] / rp_ts_df_3['size']
    bsky_df_4['deepth_normalized'] = bsky_df_4['max_depth'] / bsky_df_4['size']
    ts_df_4['deepth_normalized'] = ts_df_4['max_depth'] / ts_df_4['size']
    rp_bsky_df_4['deepth_normalized'] = rp_bsky_df_4['max_depth'] / rp_bsky_df_4['size']
    rp_ts_df_3['deepth_normalized'] = rp_ts_df_3['max_depth'] / rp_ts_df_3['size']
    return


@app.cell
def _(df_3):
    sum(df_3['size'] == 1)
    return


@app.cell
def _(df_rp):
    sum(df_rp["size"] == 1)
    return


@app.cell
def _(bsky_df_4, ks_2samp, np, pd, plt, ts_df_4):
    df_4 = pd.concat([bsky_df_4, ts_df_4], ignore_index=True)
    df_4 = df_4.loc[df_4['size'] > 1]
    ks_results_7 = []
    metrics_3 = ['bredth_normalized', 'deepth_normalized']
    for metric_22 in metrics_3:
        bsky_vals_7 = df_4[df_4['platform'] == 'bsky'][metric_22].dropna()
        ts_vals_5 = df_4[df_4['platform'] == 'ts'][metric_22].dropna()
        (ks_stat_10, p_value_11) = ks_2samp(bsky_vals_7, ts_vals_5)
        ks_results_7.append({'Metric': metric_22, 'KS Statistic': ks_stat_10, 'P-value': p_value_11})
    ks_overall_df_4 = pd.DataFrame(ks_results_7)
    (fig_27, axes_27) = plt.subplots(1, 2, figsize=(18, 6))
    for (idx_24, metric_22) in enumerate(['bredth_normalized', 'deepth_normalized']):
        for platform_17 in ['bsky', 'ts']:
            values_7 = df_4[df_4['platform'] == platform_17][metric_22].dropna()
            sorted_vals_9 = np.sort(values_7)
            ccdf_8 = 1 - np.arange(1, len(sorted_vals_9) + 1) / len(sorted_vals_9)
            axes_27[idx_24].plot(sorted_vals_9, ccdf_8, label=f'{platform_17}', linewidth=2)
        axes_27[idx_24].set_yscale('log')
        axes_27[idx_24].set_xlabel(metric_22)
        axes_27[idx_24].set_ylabel('CCDF (%)')
        axes_27[idx_24].set_title(f'CCDF of {metric_22} by Platform')
        axes_27[idx_24].grid()
        (ks_stat_10, p_value_11) = ks_overall_df_4.loc[ks_overall_df_4['Metric'] == metric_22, ['KS Statistic', 'P-value']].values[0]
        axes_27[idx_24].text(0.6, 0.1, f'KS={ks_stat_10:.4f}\nP={p_value_11:.4f}', transform=axes_27[idx_24].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.show()
    return (df_4,)


@app.cell
def _(ks_2samp, np, pd, plt, rp_bsky_df_4, rp_ts_df_3):
    df_rp = pd.concat([rp_bsky_df_4, rp_ts_df_3], ignore_index=True)
    df_rp = df_rp.loc[df_rp['size'] > 1]
    ks_results_8 = []
    metrics_4 = ['bredth_normalized', 'deepth_normalized']
    for metric_23 in metrics_4:
        bsky_vals_8 = df_rp[df_rp['platform'] == 'bsky'][metric_23].dropna()
        ts_vals_6 = df_rp[df_rp['platform'] == 'ts'][metric_23].dropna()
        (ks_stat_11, p_value_12) = ks_2samp(bsky_vals_8, ts_vals_6)
        ks_results_8.append({'Metric': metric_23, 'KS Statistic': ks_stat_11, 'P-value': p_value_12})
    ks_overall_df_5 = pd.DataFrame(ks_results_8)
    (fig_28, axes_28) = plt.subplots(1, 2, figsize=(18, 6))
    for (idx_25, metric_23) in enumerate(['bredth_normalized', 'deepth_normalized']):
        for platform_18 in ['bsky', 'ts']:
            values_8 = df_rp[df_rp['platform'] == platform_18][metric_23].dropna()
            sorted_vals_10 = np.sort(values_8)
            ccdf_9 = 1 - np.arange(1, len(sorted_vals_10) + 1) / len(sorted_vals_10)
            axes_28[idx_25].plot(sorted_vals_10, ccdf_9, label=f'{platform_18}', linewidth=2)
        axes_28[idx_25].set_yscale('log')
        axes_28[idx_25].set_xlabel(metric_23)
        axes_28[idx_25].set_ylabel('CCDF (%)')
        axes_28[idx_25].set_title(f'CCDF of {metric_23} by Platform')
        axes_28[idx_25].grid()
        (ks_stat_11, p_value_12) = ks_overall_df_5.loc[ks_overall_df_5['Metric'] == metric_23, ['KS Statistic', 'P-value']].values[0]
        axes_28[idx_25].text(0.6, 0.1, f'KS={ks_stat_11:.4f}\nP={p_value_12:.4f}', transform=axes_28[idx_25].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.show()
    return (df_rp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# F. Null models with Z scores""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Reply Network""")
    return


@app.cell
def _(df_4, np, plt, tqdm):
    import networkx as nx

    def compute_tree_metrics(tree):
        levels = {}
        n = len(tree.nodes())
        for node in tree.nodes():
            depth = nx.shortest_path_length(tree, source=0, target=node)
            if depth in levels:
                levels[depth] = levels[depth] + 1
            else:
                levels[depth] = 1
        breadth = max(levels.values())
        depth = max(levels.keys())
        shortest_paths = dict(nx.all_pairs_shortest_path_length(tree))
        total_distance = sum((shortest_paths[i][j] for i in tree.nodes for j in tree.nodes if i != j))
        virality = total_distance / (n * (n - 1)) if n > 1 else 0
        return (breadth, depth, virality)

    def generate_null_trees(n_nodes, num_null_samples=100):
        from tqdm.auto import tqdm
        if n_nodes == 1:
            return np.zeros((num_null_samples, 3))
        if n_nodes == 2:
            return np.zeros((num_null_samples, 3))
        null_metrics = []
        for _ in range(num_null_samples):
            prufer = np.random.randint(0, n_nodes, n_nodes - 2)
            tree = nx.Graph(nx.from_prufer_sequence(prufer))
            (breadth, depth, virality) = compute_tree_metrics(tree)
            null_metrics.append((breadth, depth, virality))
        return np.array(null_metrics)

    def compute_z_scores(observed_value, null_values):
        mu = np.mean(null_values)
        sigma = np.std(null_values)
        return (observed_value - mu) / sigma if sigma != 0 else 0
    z_scores_breadth = []
    z_scores_depth = []
    z_scores_virality = []
    for (_, row_2) in tqdm(df_4.iterrows(), total=len(df_4), desc='Computing Z-scores'):
        n_i = int(row_2['size'])
        B_i = row_2['breadth']
        D_i = row_2['max_depth']
        V_i = row_2['structural_virality'] if row_2['size'] > 1 else 0
        null_metrics = generate_null_trees(n_i)
        null_breadths = null_metrics[:, 0]
        null_depths = null_metrics[:, 1]
        null_virality = null_metrics[:, 2]
        Z_B_i = compute_z_scores(B_i, null_breadths)
        Z_D_i = compute_z_scores(D_i, null_depths)
        Z_V_i = compute_z_scores(V_i, null_virality)
        z_scores_breadth.append(Z_B_i)
        z_scores_depth.append(Z_D_i)
        z_scores_virality.append(Z_V_i)
    df_4['Z_breadth'] = z_scores_breadth
    df_4['Z_depth'] = z_scores_depth
    df_4['Z_virality'] = z_scores_virality
    df_4.to_csv('z_score_results.csv', index=False)

    def plot_ccdf(data, title, xlabel, platform=None):
        sorted_data = np.sort(data)
        ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.figure(figsize=(6, 4))
        plt.plot(sorted_data, ccdf, marker='o', linestyle='-', markersize=4, label=platform if platform else 'All Platforms')
        plt.xlabel(xlabel)
        plt.ylabel('CCDF')
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.show()
    for platform_19 in df_4['platform'].unique():
        df_platform = df_4[df_4['platform'] == platform_19]
        plot_ccdf(df_platform['Z_breadth'], f'CCDF of Z-scores (Breadth) - {platform_19}', 'Z-score (Breadth)', platform_19)
        plot_ccdf(df_platform['Z_depth'], f'CCDF of Z-scores (Depth) - {platform_19}', 'Z-score (Depth)', platform_19)
        plot_ccdf(df_platform['Z_virality'], f'CCDF of Z-scores (Virality) - {platform_19}', 'Z-score (Virality)', platform_19)
    return df_platform, null_metrics


@app.cell
def _(null_metrics):
    null_metrics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# G. Empirical distribution""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Reply""")
    return


@app.cell
def _(bsky_df_4, np, rp_bsky_df_4, rp_ts_df_3, ts_df_4):
    bsky_df_4['log_size'] = np.log10(bsky_df_4['size'])
    ts_df_4['log_size'] = np.log10(ts_df_4['size'])
    rp_bsky_df_4['log_size'] = np.log10(rp_bsky_df_4['size'])
    rp_ts_df_3['log_size'] = np.log10(rp_ts_df_3['size'])
    return


@app.cell
def _(bsky_df_4, handles_8, labels_8, np, plt, sns, ts_df_4):
    ts_df_4['max_depth_size_ratio'] = ts_df_4['max_depth'] / ts_df_4['size']
    unique_partisanship_1 = list(set(ts_df_4['partisanship'].unique()) | set(bsky_df_4['partisanship'].unique()))
    shared_palette_1 = sns.color_palette('tab10', len(unique_partisanship_1))
    palette_dict_1 = dict(zip(unique_partisanship_1, shared_palette_1))
    (fig_29, axes_29) = plt.subplots(1, 2, figsize=(14, 6))
    ts_df_4['log_size'] = np.log10(ts_df_4['size'])
    bsky_df_4['log_size'] = np.log10(bsky_df_4['size'])
    ts_df_4['log_max_depth'] = np.log10(ts_df_4['max_depth'])
    bsky_df_4['log_max_depth'] = np.log10(bsky_df_4['max_depth'])
    for (party, color) in palette_dict_1.items():
        party_data = ts_df_4[ts_df_4['partisanship'] == party]
        sns.regplot(data=party_data, x='log_size', y='log_max_depth', ax=axes_29[0], color=color, label=party)
    for (party, color) in palette_dict_1.items():
        party_data = bsky_df_4[bsky_df_4['partisanship'] == party]
        sns.regplot(data=party_data, x='log_size', y='log_max_depth', ax=axes_29[1], color=color, label=party)
    for ax_2 in axes_29:
        ax_2.set_xlabel('Size')
        ax_2.set_ylabel('Max Depth')
    axes_29[0].set_title('bsky')
    axes_29[1].set_title('ts')
    fig_29.legend(handles_8, labels_8, title='Partisanship', loc='upper center', ncol=len(labels_8), bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(bsky_df_4, np, pd, plt, sns, ts_df_4):
    partisan_colors_2 = {'left': '#436685', 'center': '#bbcd78', 'right': '#8a2520'}

    def expected_depth(df):
        df['log_size'] = np.log10(df['size'].replace(0, np.nan))
        df['log_depth'] = np.log10(df['max_depth'].replace(0, np.nan))
        df['size_bin'] = pd.qcut(df['log_size'], q=10, duplicates='drop')
        depth_median = df.groupby('size_bin')['log_depth'].median()
        df['expected_depth'] = df['size_bin'].map(depth_median).astype(float)
        return df
    bsky_df_5 = expected_depth(bsky_df_4)
    ts_df_5 = expected_depth(ts_df_4)

    def identify_outliers(df):
        return df[(df['size'] > 100) & (df['log_depth'] < df['expected_depth'] - 0.5)]
    bsky_outliers = identify_outliers(bsky_df_5)
    ts_outliers_1 = identify_outliers(ts_df_5)
    (fig_30, axes_30) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    sns.scatterplot(data=bsky_df_5, x='size', y='max_depth', ax=axes_30[0], color='gray', alpha=0.3)
    sns.scatterplot(data=ts_df_5, x='size', y='max_depth', ax=axes_30[1], color='gray', alpha=0.3)

    def plot_outliers_with_colors(data, ax):
        for (partisanship, color) in partisan_colors_2.items():
            subset = data[data['partisanship'] == partisanship]
            sns.scatterplot(data=subset, x='size', y='max_depth', ax=ax, color=color, label=partisanship.capitalize(), s=50, edgecolor='black', alpha=0.8)
    plot_outliers_with_colors(bsky_outliers, axes_30[0])
    plot_outliers_with_colors(ts_outliers_1, axes_30[1])
    for ax_3 in axes_30:
        ax_3.set(xscale='log', yscale='log')
        ax_3.legend(title='Partisanship (Outliers)', loc='upper left')
        ax_3.set_xlabel('Size')
        ax_3.set_ylabel('Depth')
    axes_30[0].set_title('BlueSky: Size vs Depth (Outliers Highlighted)')
    axes_30[1].set_title('TruthSocial: Size vs Depth (Outliers Highlighted)')
    plt.tight_layout()
    plt.show()
    return bsky_df_5, ts_df_5


@app.cell
def _(bsky_df_5, plt, sns, ts_df_5):
    partisan_colors_3 = {'left': '#436685', 'center': '#bbcd78', 'right': '#8a2520'}
    ts_df_5['breadth_size_ratio'] = ts_df_5['breadth'] / ts_df_5['size']
    bsky_df_5['breadth_size_ratio'] = bsky_df_5['breadth'] / bsky_df_5['size']
    bsky_df_5['outlier'] = 'no'
    bsky_df_5.loc[(bsky_df_5['size'] > 100) & (bsky_df_5['breadth_size_ratio'] < 0.1), 'outlier'] = 'yes'
    ts_df_5['outlier'] = 'no'
    ts_df_5.loc[(ts_df_5['size'] > 1000) & (ts_df_5['breadth_size_ratio'] < 0.1), 'outlier'] = 'yes'
    (fig_31, axes_31) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    sns.scatterplot(data=bsky_df_5[bsky_df_5['outlier'] == 'no'], x='size', y='breadth', ax=axes_31[0], color='gray', alpha=0.3)
    sns.scatterplot(data=ts_df_5[ts_df_5['outlier'] == 'no'], x='size', y='breadth', ax=axes_31[1], color='gray', alpha=0.3)

    def plot_outliers_with_colors_1(data, ax):
        for (partisanship, color) in partisan_colors_3.items():
            subset = data[data['partisanship'] == partisanship]
            sns.scatterplot(data=subset, x='size', y='breadth', ax=ax, color=color, label=partisanship.capitalize(), s=50, edgecolor='black', alpha=0.8)
    plot_outliers_with_colors_1(bsky_df_5[bsky_df_5['outlier'] == 'yes'], axes_31[0])
    plot_outliers_with_colors_1(ts_df_5[ts_df_5['outlier'] == 'yes'], axes_31[1])
    for ax_4 in axes_31:
        ax_4.set(xscale='log', yscale='log')
        ax_4.set_xlabel('Size')
        ax_4.set_ylabel('Breadth')
    (handles_9, labels_9) = axes_31[1].get_legend_handles_labels()
    axes_31[0].get_legend().remove()
    axes_31[1].get_legend().remove()
    fig_31.legend(handles_9, labels_9, title='Partisanship (Outliers)', loc='upper center', ncol=len(labels_9), bbox_to_anchor=(0.5, 1.05))
    axes_31[0].set_title('BlueSky: Size vs Breadth (Outliers Highlighted)')
    axes_31[1].set_title('TruthSocial: Size vs Breadth (Outliers Highlighted)')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Size vs. Depth and Breadth""")
    return


@app.cell
def _(bsky_df_5, f_1, np, pd, plt, ts_df_5):
    import statsmodels.api as sm
    from scipy.stats import f
    colors_4 = {'bsky_scatter': '#007FFF', 'bsky_rlm': '#0056A3', 'ts_scatter': '#FFD700', 'ts_rlm': '#C49A00'}
    df_5 = pd.concat([bsky_df_5, ts_df_5], ignore_index=True)
    df_bsky_1 = df_5[df_5['platform'] == 'bsky'].copy()
    df_ts_1 = df_5[df_5['platform'] == 'ts'].copy()
    df_bsky_1['log_size'] = np.log10(df_bsky_1['size'])
    df_bsky_1['log_breadth'] = np.log10(df_bsky_1['breadth'])
    df_ts_1['log_size'] = np.log10(df_ts_1['size'])
    df_ts_1['log_breadth'] = np.log10(df_ts_1['breadth'])

    def robust_fit(x, y):
        """Fits a robust regression model and returns sorted predictions."""
        x = sm.add_constant(x)
        model = sm.RLM(y, x, M=sm.robust.norms.HuberT())
        results = model.fit()
        x_sorted = np.sort(x[:, 1])
        x_pred = sm.add_constant(x_sorted)
        y_pred = results.predict(x_pred)
        print(results.summary())
        return (x_sorted, y_pred, results)
    (bsky_x, bsky_y, model_bsky) = robust_fit(df_bsky_1['log_size'].values, df_bsky_1['log_breadth'].values)
    (ts_x, ts_y, model_ts) = robust_fit(df_ts_1['log_size'].values, df_ts_1['log_breadth'].values)
    bsky_slope = model_bsky.params[1]
    ts_slope = model_ts.params[1]
    bsky_std = model_bsky.bse[1]
    ts_std = model_ts.bse[1]
    print(f'BlueSky Slope: {bsky_slope:.4f} ± {bsky_std:.4f}')
    print(f'TruthSocial Slope: {ts_slope:.4f} ± {ts_std:.4f}')

    def chow_test(x1, y1, x2, y2):
        """Performs the Chow test to compare regression slopes between two datasets."""
        (x1_const, x2_const) = (sm.add_constant(x1), sm.add_constant(x2))
        model1 = sm.RLM(y1, x1_const, M=sm.robust.norms.HuberT()).fit()
        model2 = sm.RLM(y2, x2_const, M=sm.robust.norms.HuberT()).fit()
        x_combined = np.concatenate([x1, x2])
        y_combined = np.concatenate([y1, y2])
        x_combined_const = sm.add_constant(x_combined)
        model_combined = sm.RLM(y_combined, x_combined_const, M=sm.robust.norms.HuberT()).fit()
        SSR_combined = np.sum(model_combined.resid ** 2)
        SSR1 = np.sum(model1.resid ** 2)
        SSR2 = np.sum(model2.resid ** 2)
        (n1, n2) = (len(y1), len(y2))
        k = 2
        chow_stat = (SSR_combined - (SSR1 + SSR2)) / k / ((SSR1 + SSR2) / (n1 + n2 - 2 * k))
        p_value = 1 - f_1.cdf(chow_stat, k, n1 + n2 - 2 * k)
        return (chow_stat, p_value)
    (chow_stat, p_value_13) = chow_test(df_bsky_1['log_size'].values, df_bsky_1['log_breadth'].values, df_ts_1['log_size'].values, df_ts_1['log_breadth'].values)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(df_bsky_1['log_size'], df_bsky_1['log_breadth'], alpha=0.3, color=colors_4['bsky_scatter'], label='BlueSky Data')
    plt.scatter(df_ts_1['log_size'], df_ts_1['log_breadth'], alpha=0.3, color=colors_4['ts_scatter'], label='TruthSocial Data')
    plt.plot(bsky_x, bsky_y, color=colors_4['bsky_rlm'], linewidth=2, label='BlueSky Robust Fit')
    plt.plot(ts_x, ts_y, color=colors_4['ts_rlm'], linewidth=2, label='TruthSocial Robust Fit')
    plt.xlabel('Size)')
    plt.ylabel('Breadth')
    plt.title('Robust Regression of Breadth Across Platforms in Reply Network')
    plt.text(min(df_bsky_1['log_size']), max(df_bsky_1['log_breadth']), f'BlueSky Slope: {bsky_slope:.4f}\nTruthSocial Slope: {ts_slope:.4f}', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return df_5, sm


@app.cell
def _(df_5, np, pd, plt, sm):
    colors_5 = {'bsky': '#007FFF', 'ts': '#FFD700'}
    low_thresh = 0.4
    high_thresh = 0.6
    df_5['alignment_category'] = pd.cut(df_5['alignment_ratio'], bins=[-np.inf, low_thresh, high_thresh, np.inf], labels=['Low', 'Medium', 'High'])
    df_5['log_size'] = np.log10(df_5['size'])
    df_5['log_breadth'] = np.log10(df_5['breadth'])

    def robust_fit_1(x, y):
        x = sm.add_constant(x)
        model = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
        x_sorted = np.sort(x[:, 1])
        x_pred = sm.add_constant(x_sorted)
        y_pred = model.predict(x_pred)
        return (x_sorted, y_pred, model)
    plt.figure(figsize=(12, 8), dpi=300)
    for platform_20 in ['bsky', 'ts']:
        for category in ['Low', 'Medium', 'High']:
            sub_df = df_5[(df_5['platform'] == platform_20) & (df_5['alignment_category'] == category)]
            if len(sub_df) < 5:
                continue
            x_vals = sub_df['log_size'].values
            y_vals = sub_df['log_breadth'].values
            (x_fit, y_fit, model) = robust_fit_1(x_vals, y_vals)
            slope = model.params[1]
            stderr = model.bse[1]
            print(f'{platform_20.upper()} - {category}: β = {slope:.4f}; Standard Error: {stderr:.4f}')
            plt.scatter(x_vals, y_vals, alpha=0.3, color=colors_5[platform_20])
            linestyle = '-' if category == 'Low' else '--' if category == 'Medium' else ':'
            plt.plot(x_fit, y_fit, linewidth=2, linestyle=linestyle, color=colors_5[platform_20], label=f'{platform_20.upper()} RLM - {category}')
            label_x = x_fit[-1]
            label_y = y_fit[-1]
            plt.text(label_x, label_y, f'β={slope:.2f}', fontsize=9, ha='left', va='center', color=colors_5[platform_20], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.xlabel('Log(Size)')
    plt.ylabel('Log(Breadth)')
    plt.title('Robust Regression of Breadth by Platform and Alignment Category')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Figure 2:""")
    return


@app.cell
def _(bsky_df_5, np, pd, plt, rp_bsky_df_4, rp_ts_df_3, sm, ts_df_5):
    colors_6 = {'bsky': '#5F9EA0', 'ts': '#FF6347'}
    df_reply = pd.concat([bsky_df_5, ts_df_5], ignore_index=True)
    df_reply = df_reply[(df_reply['size'] > 0) & (df_reply['breadth'] > 0) & (df_reply['max_depth'] >= 0)].copy()
    df_reply['log_size'] = np.log10(df_reply['size'])
    df_reply['log_breadth'] = np.log10(df_reply['breadth'])
    df_reply['log_depth'] = np.log10(df_reply['max_depth'] + 1)
    df_reply['weight'] = 1 / df_reply['size']
    df_repost = pd.concat([rp_bsky_df_4, rp_ts_df_3], ignore_index=True)
    df_repost = df_repost[(df_repost['size'] > 0) & (df_repost['breadth'] > 0) & (df_repost['max_depth'] >= 0)].copy()
    df_repost['log_size'] = np.log10(df_repost['size'])
    df_repost['log_breadth'] = np.log10(df_repost['breadth'])
    df_repost['log_depth'] = np.log10(df_repost['max_depth'] + 1)
    df_repost['weight'] = df_repost['size']

    def wls_fit(x, y, weights):
        X = sm.add_constant(x)
        model = sm.RLM(y, X, weights=weights, M=sm.robust.norms.HuberT()).fit()
        x_sorted = np.sort(x)
        X_pred = sm.add_constant(x_sorted)
        y_pred = model.predict(X_pred)
        return (x_sorted, y_pred, model)
    (fig_32, axes_32) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    metrics_5 = ['log_breadth', 'log_depth']
    titles = ['Cascade Max Breadth', 'Cascade Depth']
    for (i_8, (y_var, title)) in enumerate(zip(metrics_5, titles)):
        ax_5 = axes_32[0, i_8]
        for platform_21 in ['bsky', 'ts']:
            df_6 = df_reply[df_reply['platform'] == platform_21]
            x = df_6['log_size']
            y = df_6[y_var]
            w = df_6['weight']
            (x_fit_1, y_fit_1, model_1) = wls_fit(x, y, w)
            marker = 'o' if platform_21 == 'bsky' else 's'
            ax_5.scatter(x, y, alpha=0.07, color=colors_6[platform_21], marker=marker, edgecolors='white', linewidths=1.5, label=None, zorder=3 if platform_21 == 'bsky' else 2)
            ax_5.plot(x_fit_1, y_fit_1, color=colors_6[platform_21], linewidth=3, label='BlueSky' if platform_21 == 'bsky' else 'TruthSocial', zorder=10)
            platform_label = 'BlueSky' if platform_21 == 'bsky' else 'TruthSocial'
            ax_5.text(0.02, 0.95 - 0.08 * (platform_21 == 'ts'), f'{platform_label} Slope: {model_1.params[1]:.4f} ± {model_1.bse[1]:.4f}', transform=ax_5.transAxes, fontsize=16, ha='left', va='top')
        ax_5.set_ylabel(title, fontsize=20, fontweight='bold')
        ax_5.tick_params(labelsize=18)
        ax_5.spines['top'].set_visible(False)
        ax_5.spines['right'].set_visible(False)
    for (i_8, (y_var, title)) in enumerate(zip(metrics_5, titles)):
        ax_5 = axes_32[1, i_8]
        for platform_21 in ['bsky', 'ts']:
            df_6 = df_repost[df_repost['platform'] == platform_21]
            x = df_6['log_size']
            y = df_6[y_var]
            w = df_6['weight']
            (x_fit_1, y_fit_1, model_1) = wls_fit(x, y, w)
            marker = 'o' if platform_21 == 'bsky' else 's'
            ax_5.scatter(x, y, alpha=0.05, color=colors_6[platform_21], marker=marker, edgecolors='white', linewidths=1.5, label=None, zorder=3 if platform_21 == 'bsky' else 2)
            ax_5.plot(x_fit_1, y_fit_1, color=colors_6[platform_21], linewidth=3, label='BlueSky' if platform_21 == 'bsky' else 'TruthSocial', zorder=10)
            platform_label = 'BlueSky' if platform_21 == 'bsky' else 'TruthSocial'
            ax_5.text(0.02, 0.95 - 0.08 * (platform_21 == 'ts'), f'{platform_label} Slope: {model_1.params[1]:.4f} ± {model_1.bse[1]:.4f}', transform=ax_5.transAxes, fontsize=16, ha='left', va='top')
        ax_5.set_xlabel('Cascade Size', fontsize=20, fontweight='bold')
        ax_5.set_ylabel(title, fontsize=20, fontweight='bold')
        ax_5.tick_params(labelsize=18)
        ax_5.spines['top'].set_visible(False)
        ax_5.spines['right'].set_visible(False)
    (handles_10, labels_10) = axes_32[0, 0].get_legend_handles_labels()
    fig_32.legend(handles_10, labels_10, loc='upper center', bbox_to_anchor=(0.14, 0.84), ncol=1, frameon=False, fontsize=14)
    subplot_labels_1 = ['(A)', '(B)', '(C)', '(D)']
    for (ax_5, label_5) in zip(axes_32.flat, subplot_labels_1):
        ax_5.text(0.9, 1, label_5, transform=ax_5.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    return (df_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Model it by bin""")
    return


@app.cell
def _(df_6, np, pd, sm):
    bin_width = 0.1
    df_6.loc[df_6['size'] == 1, 'alignment_ratio'] = 0
    bins = np.arange(0, 1 + bin_width, bin_width)
    bin_labels = [f'{round(b, 2)}–{round(b + bin_width, 2)}' for b in bins[:-1]]
    df_6['alignment_bin'] = pd.cut(df_6['alignment_ratio'], bins=bins, labels=bin_labels, include_lowest=True)
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_depth'] = np.log10(df_6['breadth'])

    def robust_fit_2(x, y):
        x = sm.add_constant(x)
        model = sm.OLS(y, x, M=sm.robust.norms.HuberT()).fit()
        return model
    results = []
    for platform_22 in ['bsky', 'ts']:
        for bin_label in bin_labels:
            sub_df_1 = df_6[(df_6['platform'] == platform_22) & (df_6['alignment_bin'] == bin_label)]
            if len(sub_df_1) < 5:
                continue
            x_vals_1 = sub_df_1['log_size'].values
            y_vals_1 = sub_df_1['log_depth'].values
            model_2 = robust_fit_2(x_vals_1, y_vals_1)
            slope_1 = model_2.params[1]
            stderr_1 = model_2.bse[1]
            results.append({'Platform': platform_22, 'Alignment Bin': bin_label, 'Slope (β)': slope_1, 'Std Error': stderr_1, 'N': len(sub_df_1)})
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['Alignment Bin', 'Platform'])
    return (results_df,)


@app.cell
def _(results_df):
    results_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(df_6):
    df_6.log
    return


@app.cell
def _(df_6, np, pd, plt, sm):
    bin_width_1 = 0.1
    bins_1 = np.arange(0, 1 + bin_width_1, bin_width_1)
    bin_labels_1 = [f'{round(b, 2)}–{round(b + bin_width_1, 2)}' for b in bins_1[:-1]]
    df_6['alignment_bin'] = pd.cut(df_6['alignment_ratio'], bins=bins_1, labels=bin_labels_1, include_lowest=True)
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_depth'] = np.log10(df_6['max_depth'] + 1)

    def robust_fit_3(x, y):
        x = sm.add_constant(x)
        model = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
        return model
    target_bins = ['0.0–0.1', '0.9–1.0']
    results_1 = []
    x_axis = 0
    y_axis = 0
    (fit, ax_6) = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    for platform_23 in ['bsky', 'ts']:
        for bin_label_1 in target_bins:
            sub_df_2 = df_6[(df_6['platform'] == platform_23) & (df_6['alignment_bin'] == bin_label_1)]
            if len(sub_df_2) < 5:
                continue
            x_vals_2 = sub_df_2['log_size'].values
            y_vals_2 = sub_df_2['log_depth'].values
            model_3 = robust_fit_3(x_vals_2, y_vals_2)
            slope_2 = model_3.params[1]
            intercept = model_3.params[0]
            stderr_2 = model_3.bse[1]
            results_1.append({'Platform': platform_23, 'Alignment Bin': bin_label_1, 'Slope (β)': slope_2, 'Intercept': intercept, 'Std Error': stderr_2, 'N': len(sub_df_2)})
            x_line = np.linspace(x_vals_2.min(), x_vals_2.max(), 100)
            y_line = intercept + slope_2 * x_line
            ax_6[x_axis, y_axis].scatter(x_vals_2, y_vals_2, alpha=0.6, label='Data')
            ax_6[x_axis, y_axis].plot(x_line, y_line, color='red', label='Robust Fit')
            model_3 = sm.OLS(y_vals_2, sm.add_constant(x_vals_2)).fit()
            y_pred = model_3.predict(sm.add_constant(x_line))
            ax_6[x_axis, y_axis].plot(x_line, y_pred, color='blue', linestyle='--', label='Linear Fit')
            ax_6[x_axis, y_axis].set_title(f'Robust Fit for {platform_23.upper()} — Bin {bin_label_1}')
            ax_6[x_axis, y_axis].set_xlabel('log10(Size)')
            ax_6[x_axis, y_axis].set_ylabel('log10(Depth + 1)')
            ax_6[x_axis, y_axis].legend()
            ax_6[x_axis, y_axis].grid(True)
            if y_axis == 1:
                y_axis = 0
                x_axis = x_axis + 1
            else:
                y_axis = y_axis + 1
    return


@app.cell
def _(df_6, np, sm):
    import statsmodels.formula.api as smf
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_breadth'] = np.log10(df_6['breadth'])
    df_6['platform_ts'] = (df_6['platform'] == 'ts').astype(int)
    X = df_6[['log_size', 'alignment_ratio', 'platform_ts']].copy()
    X['log_size:alignment_ratio'] = X['log_size'] * X['alignment_ratio']
    X = sm.add_constant(X)
    y_1 = df_6['log_breadth']
    model_4 = sm.RLM(y_1, X, M=sm.robust.norms.HuberT()).fit()
    print(model_4.summary())
    return (smf,)


@app.cell
def _(df_6, pd, plt, sm, smf, sns):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey, linear_reset
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from scipy import stats
    df_6['platform'] = df_6['platform'].astype('category')
    model_5 = smf.ols('log_breadth ~ log_size *  platform', data=df_6).fit(cov_type='HC3')
    print(model_5.summary())
    X_1 = model_5.model.exog
    vif = pd.DataFrame()
    vif['Variable'] = model_5.model.exog_names
    vif['VIF'] = [variance_inflation_factor(X_1, i) for i in range(X_1.shape[1])]
    print('\nVIF 检验结果：')
    print(vif)
    bp_test = het_breuschpagan(model_5.resid, model_5.model.exog)
    labels_11 = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    print('\nBreusch-Pagan 检验结果：')
    print(dict(zip(labels_11, bp_test)))
    dw = durbin_watson(model_5.resid)
    print(f'\nDurbin-Watson 检验结果： {dw:.3f}')
    (jb_stat, jb_pvalue, _, _) = jarque_bera(model_5.resid)
    print(f'\nJarque-Bera 检验：Statistic={jb_stat:.3f}, p-value={jb_pvalue:.3f}')
    sm.qqplot(model_5.resid, line='45')
    plt.title('QQ Plot of Residuals')
    plt.show()
    reset_test = linear_reset(model_5, power=2, use_f=True)
    print('\nRamsey RESET 检验结果：')
    print(reset_test)
    plt.figure()
    sns.residplot(x=model_5.fittedvalues, y=model_5.resid, lowess=True)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.axhline(0, linestyle='--', color='red')
    plt.show()
    return (
        durbin_watson,
        het_breuschpagan,
        jarque_bera,
        linear_reset,
        variance_inflation_factor,
    )


@app.cell
def _(
    df_6,
    durbin_watson,
    het_breuschpagan,
    jarque_bera,
    linear_reset,
    np,
    pd,
    plt,
    sm,
    smf,
    sns,
    variance_inflation_factor,
):
    df_6['platform'] = df_6['platform'].astype('category')
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_breadth'] = np.log10(df_6['breadth'])
    model_6 = smf.ols('log_breadth ~ log_size * platform', data=df_6).fit()
    print(model_6.summary())
    X_2 = model_6.model.exog
    vif_1 = pd.DataFrame()
    vif_1['Variable'] = model_6.model.exog_names
    vif_1['VIF'] = [variance_inflation_factor(X_2, i) for i in range(X_2.shape[1])]
    print('\nVIF 检验结果：')
    print(vif_1)
    bp_test_1 = het_breuschpagan(model_6.resid, model_6.model.exog)
    labels_12 = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    print('\nBreusch-Pagan 检验结果：')
    print(dict(zip(labels_12, bp_test_1)))
    dw_1 = durbin_watson(model_6.resid)
    print(f'\nDurbin-Watson 检验结果： {dw_1:.3f}')
    (jb_stat_1, jb_pvalue_1, _, _) = jarque_bera(model_6.resid)
    print(f'\nJarque-Bera 检验：Statistic={jb_stat_1:.3f}, p-value={jb_pvalue_1:.3f}')
    sm.qqplot(model_6.resid, line='45')
    plt.title('QQ Plot of Residuals')
    plt.show()
    reset_test_1 = linear_reset(model_6, power=2, use_f=True)
    print('\nRamsey RESET 检验结果：')
    print(reset_test_1)
    plt.figure()
    sns.residplot(x=model_6.fittedvalues, y=model_6.resid, lowess=True)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.axhline(0, linestyle='--', color='red')
    plt.show()
    return X_2, model_6


@app.cell
def _(model_6):
    robust_model = model_6.get_robustcov_results(cov_type='HC3')
    print(robust_model.summary())
    return


@app.cell
def _(df_6, huber_model, np, plt):
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_breadth'] = np.log10(df_6['breadth'])
    weights = 1 / df_6['size'] ** 2
    print(huber_model.summary())
    from statsmodels.robust.norms import HuberT
    from statsmodels.graphics.regressionplots import plot_leverage_resid2, influence_plot
    residuals = huber_model.resid
    fitted = huber_model.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted, residuals, alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Huber Regression)')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### WLS Model breadth""")
    return


@app.cell
def _(df_6):
    df_6.loc[df_6['size'] == 1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Baseline""")
    return


@app.cell
def _(df_6, np, plt, sm, smf):
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_breadth'] = np.log10(df_6['breadth'])
    df_6.loc[df_6['size'] == 1, 'alignment_ratio'] = 1
    formula1 = 'log_breadth ~ log_size * platform'
    weights_1 = 1 / df_6['size']
    model1 = smf.rlm(formula1, data=df_6, M=sm.robust.norms.HuberT()).fit()
    print(model1.summary())
    residuals_1 = model1.resid
    fitted_1 = model1.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_1, residuals_1, alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Huber Regression)')
    plt.show()

    def evaluate_rlm(model, y_true):
        y_hat = model.fittedvalues
        resid = model.resid
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        rss = np.sum(resid ** 2)
        pseudo_r2 = 1 - rss / tss
        mae = np.mean(np.abs(resid))
        rmse = np.sqrt(np.mean(resid ** 2))
        print(f'Pseudo R²: {pseudo_r2:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'RMSE: {rmse:.4f}')
    evaluate_rlm(model1, df_6['log_breadth'])
    return formula1, model1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Add outliers""")
    return


@app.cell
def _(df_6, df_platform, np, plt, sm, smf):
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_breadth'] = np.log10(df_6['breadth'])
    weights_2 = 1 / df_6['size']
    df_6['outlier'].replace({np.nan: '0', False: '1', True: '2'}, inplace=True)
    df_6['outlier'] = df_6['outlier'].astype('category')
    formula2 = 'log_breadth ~ log_size * outlier'
    model2 = smf.rlm(formula2, data=df_platform, M=sm.robust.norms.HuberT()).fit()
    print(model2.summary())
    residuals_2 = model2.resid
    fitted_2 = model2.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_2, residuals_2, alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Huber Regression)')
    plt.show()

    def evaluate_rlm_1(model, y_true):
        y_hat = model.fittedvalues
        resid = model.resid
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        rss = np.sum(resid ** 2)
        pseudo_r2 = 1 - rss / tss
        mae = np.mean(np.abs(resid))
        rmse = np.sqrt(np.mean(resid ** 2))
        print(f'Pseudo R²: {pseudo_r2:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'RMSE: {rmse:.4f}')
    evaluate_rlm_1(model2, df_6['log_breadth'])
    return formula2, model2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Add alignment ratio""")
    return


@app.cell
def _():
    ##### Compare whether to add three way interaction
    return


@app.cell
def _(df_6, sm, smf):
    model_2way = smf.rlm('log_breadth ~ log_size * C(platform) + log_size * cr(alignment_ratio, df=20)', data=df_6, weights=1 / df_6['size'], M=sm.robust.norms.HuberT()).fit()
    model_3way = smf.rlm('log_breadth ~ log_size * C(platform) * cr(alignment_ratio, df=5)', data=df_6, weights=1 / df_6['size'], M=sm.robust.norms.HuberT()).fit()
    from statsmodels.stats.anova import anova_lm
    anova_results = anova_lm(model_2way, model_3way)
    print(anova_results)
    return model_2way, model_3way


@app.cell
def _(model_2way, model_3way):
    from statsmodels.iolib.summary2 import summary_col

    # 输出 AIC, BIC, R²
    print("AIC 2-way:", model_2way.aic)
    print("AIC 3-way:", model_3way.aic)
    print("BIC 2-way:", model_2way.bic)
    print("BIC 3-way:", model_3way.bic)
    print("R² 2-way:", model_2way.rsquared)
    print("R² 3-way:", model_3way.rsquared)
    return


@app.cell
def _(df_6, np, pd, plt, sm, smf):
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_breadth'] = np.log10(df_6['breadth'])
    weights_3 = 1 / df_6['size']
    formula3 = 'log_breadth ~ log_size * platform   + log_size * cr(alignment_ratio,df=8)'
    model3 = smf.rlm(formula3, data=df_6, M=sm.robust.norms.HuberT()).fit()
    print(model3.summary())
    residuals_3 = model3.resid
    fitted_3 = model3.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_3, residuals_3, alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Huber Regression)')
    plt.show()
    from patsy import dmatrix
    alignment_range = np.linspace(df_6['alignment_ratio'].min(), df_6['alignment_ratio'].max(), 100)
    size_levels = np.linspace(df_6['log_size'].min(), df_6['log_size'].max(), 10)
    plot_data = []
    for size_val in size_levels:
        temp_df = pd.DataFrame({'alignment_ratio': alignment_range, 'log_size': size_val, 'platform': df_6['platform'].mode()[0]})
        temp_df['log_breadth_pred'] = model3.predict(temp_df)
        temp_df['log_size_level'] = f'log_size = {size_val:.2f}'
        plot_data.append(temp_df)
    plot_df = pd.concat(plot_data)
    plt.figure(figsize=(10, 6))
    for (name, group) in plot_df.groupby('log_size_level'):
        plt.plot(group['alignment_ratio'], group['log_breadth_pred'], label=name)
    plt.xlabel('Alignment Ratio')
    plt.ylabel('Predicted Log Breadth')
    plt.title('Marginal Effect of Alignment Ratio (by Log Size Level)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    def evaluate_rlm_2(model, y_true):
        y_hat = model.fittedvalues
        resid = model.resid
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        rss = np.sum(resid ** 2)
        pseudo_r2 = 1 - rss / tss
        mae = np.mean(np.abs(resid))
        rmse = np.sqrt(np.mean(resid ** 2))
        print(f'Pseudo R²: {pseudo_r2:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'RMSE: {rmse:.4f}')
    print('Evaluate RLM Model:')
    evaluate_rlm_2(model3, df_6['log_breadth'])
    return dmatrix, evaluate_rlm_2, formula3, model3, weights_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Add topic""")
    return


@app.cell
def _(df_6, evaluate_rlm_2, np, plt, sm, smf, weights_3):
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_breadth'] = np.log10(df_6['breadth'])
    formula4 = 'log_breadth ~ log_size * platform + log_size * topic_label'
    model4 = smf.rlm(formula4, data=df_6, M=sm.robust.norms.HuberT(), weights=weights_3).fit()
    print(model4.summary())
    residuals_4 = model4.resid
    fitted_4 = model4.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_4, residuals_4, alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Huber Regression)')
    plt.show()
    evaluate_rlm_2(model4, df_6['log_breadth'])
    return formula4, model4


@app.cell
def _(df_6):
    df_6.to_csv('df.csv', index=False)
    return


@app.cell
def _(model1, model2, model3, model4):
    lr_stat = 2 * (model2.llf - model1.llf)
    df_diff = model2.df_model - model1.df_model
    from scipy.stats import chi2
    p_value_14 = chi2.sf(lr_stat, df_diff)
    print(f'LR stat 1 vs 2: {lr_stat:.4f}, p-value: {p_value_14:.4f}')
    lr_stat = 2 * (model3.llf - model1.llf)
    df_diff = model3.df_model - model1.df_model
    p_value_14 = chi2.sf(lr_stat, df_diff)
    print(f'LR stat 1 vs 3: {lr_stat:.4f}, p-value: {p_value_14:.4f}')
    lr_stat = 2 * (model4.llf - model1.llf)
    df_diff = model4.df_model - model1.df_model
    p_value_14 = chi2.sf(lr_stat, df_diff)
    print(f'LR stat 1 vs 4: {lr_stat:.4f}, p-value: {p_value_14:.4f}')
    return


@app.cell
def _(df_6, formula1, formula2, formula3, formula4, np, smf):
    from sklearn.model_selection import KFold
    from scipy.stats import ttest_rel, wilcoxon

    def crossval_r2(formula, data, k=100):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        r2s = []
        for (train_idx, test_idx) in kf.split(data):
            (train, test) = (data.iloc[train_idx], data.iloc[test_idx])
            model = smf.rlm(formula, data=train).fit()
            y_true = test['log_breadth']
            y_pred = model.predict(test)
            r2 = 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
            r2s.append(r2)
        return r2s
    r2_model1 = crossval_r2(formula1, df_6)
    r2_model2 = crossval_r2(formula2, df_6)
    r2_model3 = crossval_r2(formula3, df_6)
    r2_model4 = crossval_r2(formula4, df_6)
    print('Model 1 CV R² scores:', np.mean(r2_model1))
    print('Model 2 CV R² scores:', np.mean(r2_model2))
    print('Model 3 CV R² scores:', np.mean(r2_model3))
    print('Model 4 CV R² scores:', np.mean(r2_model4))
    (t_stat, p_ttest) = ttest_rel(r2_model2, r2_model1)
    print(f'\nPaired t-test 1 vs. 2: statistic = {t_stat:.4f}, p = {p_ttest:.4f}')
    (t_stat, p_ttest) = ttest_rel(r2_model3, r2_model1)
    print(f'\nPaired t-test 3 vs. 1: statistic = {t_stat:.4f}, p = {p_ttest:.4f}')
    (t_stat, p_ttest) = ttest_rel(r2_model4, r2_model1)
    print(f'\nPaired t-test 4 vs. 1: statistic = {t_stat:.4f}, p = {p_ttest:.4f}')
    return r2_model1, r2_model2, ttest_rel


@app.cell
def _(np, r2_model1, r2_model2, r3_model3, r3_model4, ttest_rel):
    print('Model 1 CV R² scores:', np.mean(r2_model1))
    print('Model 2 CV R² scores:', np.mean(r2_model2))
    print('Model 3 CV R² scores:', np.mean(r3_model3))
    print('Model 4 CV R² scores:', np.mean(r3_model4))
    (t_stat_1, p_ttest_1) = ttest_rel(r2_model2, r2_model1)
    print(f'\nPaired t-test 1 vs. 2: statistic = {t_stat_1:.4f}, p = {p_ttest_1:.4f}')
    (t_stat_1, p_ttest_1) = ttest_rel(r3_model3, r2_model1)
    print(f'\nPaired t-test 3 vs. 1: statistic = {t_stat_1:.4f}, p = {p_ttest_1:.4f}')
    (t_stat_1, p_ttest_1) = ttest_rel(r3_model4, r2_model1)
    print(f'\nPaired t-test 4 vs. 1: statistic = {t_stat_1:.4f}, p = {p_ttest_1:.4f}')
    return


@app.cell
def _(df_6, np, plt, sm, smf):
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_depth'] = np.log10(df_6['max_depth'] + 1)
    formula1_1 = 'log_depth ~ log_size * platform'
    weights_4 = 1 / df_6['size']
    model1_1 = smf.wls(formula1_1, data=df_6, M=sm.robust.norms.HuberT(), weights=weights_4).fit()
    print(model1_1.summary())
    residuals_5 = model1_1.resid
    fitted_5 = model1_1.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_5, residuals_5, alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Huber Regression)')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### WLS model depth""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Baseline""")
    return


@app.cell
def _(df_6, evaluate_rlm_2, np, plt, sm, smf):
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_depth'] = np.log10(df_6['max_depth'] + 1)
    formula1_2 = 'log_depth ~ log_size * platform'
    weights_5 = 1 / df_6['size']
    model1_2 = smf.rlm(formula1_2, data=df_6, M=sm.robust.norms.HuberT(), weights=weights_5).fit()
    print(model1_2.summary())
    residuals_6 = model1_2.resid
    fitted_6 = model1_2.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_6, residuals_6, alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Huber Regression)')
    plt.show()
    evaluate_rlm_2(model1_2, df_6['log_depth'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Add outliers""")
    return


@app.cell
def _(df_6, evaluate_rlm_2, np, plt, sm, smf):
    weights_6 = 1 / df_6['size']
    df_6['outlier'].replace({np.nan: '0', False: '1', True: '2'}, inplace=True)
    df_6['outlier'] = df_6['outlier'].astype('category')
    formula2_1 = 'log_depth ~ log_size * outlier'
    model2_1 = smf.rlm(formula2_1, data=df_6, M=sm.robust.norms.HuberT(), weights=weights_6).fit()
    print(model2_1.summary())
    residuals_7 = model2_1.resid
    fitted_7 = model2_1.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_7, residuals_7, alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Huber Regression)')
    plt.show()
    evaluate_rlm_2(model2_1, df_6['log_depth'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Add alignment ratio""")
    return


@app.cell
def _(df_6, evaluate_rlm_2, np, pd, plt, sm, smf):
    weights_7 = 1 / df_6['size']
    formula3_1 = 'log_depth ~ log_size * platform  + log_size * cr(alignment_ratio,df=8)'
    model3_1 = smf.rlm(formula3_1, data=df_6, M=sm.robust.norms.HuberT(), weights=weights_7).fit()
    print(model3_1.summary())
    residuals_8 = model3_1.resid
    fitted_8 = model3_1.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_8, residuals_8, alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Huber Regression)')
    plt.show()
    alignment_range_1 = np.linspace(df_6['alignment_ratio'].min(), df_6['alignment_ratio'].max(), 100)
    size_levels_1 = np.linspace(df_6['log_size'].min(), df_6['log_size'].max(), 10)
    plot_data_1 = []
    for size_val_1 in size_levels_1:
        temp_df_1 = pd.DataFrame({'alignment_ratio': alignment_range_1, 'log_size': size_val_1, 'platform': df_6['platform'].mode()[0]})
        temp_df_1['log_breadth_pred'] = model3_1.predict(temp_df_1)
        temp_df_1['log_size_level'] = f'log_size = {size_val_1:.2f}'
        plot_data_1.append(temp_df_1)
    plot_df_1 = pd.concat(plot_data_1)
    plt.figure(figsize=(10, 6))
    for (name_1, group_1) in plot_df_1.groupby('log_size_level'):
        plt.plot(group_1['alignment_ratio'], group_1['log_breadth_pred'], label=name_1)
    plt.xlabel('Alignment Ratio')
    plt.ylabel('Predicted Log Breadth')
    plt.title('Marginal Effect of Alignment Ratio (by Log Size Level)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    evaluate_rlm_2(model3_1, df_6['log_depth'])
    return (weights_7,)


@app.cell
def _(df_6, np, sm, smf):
    def find_best_spline_df(data, outcome, size_var, platform_var, alignment_var, max_df=20, weight_var=None, method='aic', print_each=True):
        """
        自动寻找最优的 spline 自由度，用于非线性建模。

        参数:
            data: DataFrame
            outcome: 因变量名 (e.g., 'log_breadth')
            size_var: 自变量 log_size 名
            platform_var: 平台变量名 (e.g., 'platform')
            alignment_var: 要做样条转换的变量名 (e.g., 'alignment_ratio')
            max_df: 最大自由度（从 3 到 max_df 试）
            weight_var: WLS 的权重变量名；若为 None 则使用 OLS
            method: 'aic'（默认）或 'bic'
            print_each: 是否打印每次拟合的结果

        返回:
            best_df, best_model
        """
        best_score = np.inf
        best_df = None
        best_model = None
        df_list = []
        score_list = []
        for df_spline in range(3, max_df + 1):
            spline_terms = f'cr({alignment_var}, df={df_spline})'
            formula = f'{outcome} ~ {size_var} * {platform_var} + {size_var} * {spline_terms}'
            try:
                if weight_var:
                    weights = 1 / data[weight_var]
                    model = smf.rlm(formula=formula, data=data, weights=weights, M=sm.robust.norms.HuberT()).fit()
                else:
                    model = smf.ols(formula=formula, data=data).fit()
                y_hat = model.fittedvalues
                y_true = data[outcome]
                resid = model.resid
                tss = np.sum((y_true - np.mean(y_true)) ** 2)
                rss = np.sum(resid ** 2)
                rmse = np.sqrt(np.mean(resid ** 2))
                score = rmse
                df_list.append(df_spline)
                score_list.append(score)
                if print_each:
                    print(f'df={df_spline}: AIC={model.aic:.2f}, BIC={model.bic:.2f}')
                if score < best_score:
                    best_score = score
                    best_df = df_spline
                    best_model = model
            except Exception as e:
                print(f'df={df_spline} failed: {e}')
                continue
        print(f'\nBest df = {best_df} with {method.upper()} = {best_score:.2f}')
        return (best_df, best_model, df_list, score_list)
    (best_df, best_model, df_list, score_list) = find_best_spline_df(data=df_6, outcome='log_breadth', size_var='log_size', platform_var='platform', alignment_var='alignment_ratio', weight_var='size', method='bic')
    return df_list, score_list


@app.cell
def _(df_list, score_list):
    df_list, score_list
    return


@app.cell
def _(df_list, score_list, sns):
    #plot the df_list with score_list

    sns.set(style="whitegrid")
    sns.lineplot(x=df_list[1:], y=score_list[1:])
    return


@app.cell
def _(df_list, np, score_list):
    def detect_elbow(df_values, bic_values):
        # 标准化 df 和 bic 以避免尺度不一致
        x = np.array(df_values)
        y = np.array(bic_values)
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())

        # 拟合直线：从第一个点到最后一个点
        line = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
        line = line / np.linalg.norm(line)

        # 计算每个点到直线的垂直距离
        distances = []
        for i in range(len(x_norm)):
            vec = np.array([x_norm[i] - x_norm[0], y_norm[i] - y_norm[0]])
            proj_len = np.dot(vec, line)
            proj = proj_len * line
            dist_vec = vec - proj
            distances.append(np.linalg.norm(dist_vec))

        elbow_index = int(np.argmax(distances))
        elbow_df = df_values[elbow_index]

        return elbow_df, elbow_index, distances

    elbow_df, elbow_index, distances = detect_elbow(df_list[1:], score_list[1:])
    return elbow_df, elbow_index


@app.cell
def _(df_list, elbow_df, elbow_index, plt, score_list):
    plt.figure(figsize=(8, 5),dpi=300)
    plt.plot(df_list[1:], score_list[1:], marker='o', label='BIC', zorder=1)
    plt.scatter([df_list[elbow_index]], [score_list[elbow_index]], color='#FF9149', zorder=3,label=f"Elbow df = {elbow_df}")
    plt.xlabel("Degrees of Freedom (df)")
    plt.ylabel("BIC")
    plt.title("Elbow Detection for Spline df")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Add topic""")
    return


@app.cell
def _(df_6, evaluate_rlm_2, plt, sm, smf, weights_7):
    formula4_1 = 'log_depth ~ log_size * platform  + log_size * topic_label'
    model4_1 = smf.wls(formula4_1, data=df_6, M=sm.robust.norms.HuberT(), weights=weights_7).fit()
    print(model4_1.summary())
    residuals_9 = model4_1.resid
    fitted_9 = model4_1.fittedvalues
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_9, residuals_9, alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Huber Regression)')
    plt.show()
    evaluate_rlm_2(model4_1, df_6['log_depth'])
    return


@app.cell
def _(df_6, smf):
    formula = 'log_breadth ~ log_size * platform + log_size * cr(alignment_ratio, df=5)'
    model_7 = smf.rlm(formula, data=df_6).fit()
    print(model_7.summary())
    return


@app.cell
def _(df_6, sm, smf):
    formula_1 = 'log_breadth ~ log_size * platform + log_size * cr(alignment_ratio, df=5) + C(topic_label)*log_size'
    model_8 = smf.ols(formula_1, data=df_6, M=sm.robust.norms.HuberT()).fit()
    print(model_8.summary())
    return


@app.cell
def _(df_6):
    import bambi as bmb
    import arviz as az
    formula_2 = 'log_breadth ~ log_size * platform + log_size * bs(alignment_ratio, 5) + C(topic_label)*log_size'
    model_9 = bmb.Model(formula_2, data=df_6, family='negativebinomial', priors={'sigma': bmb.Prior('HalfNormal', sigma=1)})
    results_2 = model_9.fit(draws=1000, tune=1000)
    az.summary(results_2)
    return az, model_9


@app.cell
def _(model_9):
    model_9.backend.model.debug()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Baysian""")
    return


@app.cell
def _(X_2):
    X_2.shape
    return


@app.cell
def _(df_6, np):
    import pymc as pm
    import patsy
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,5,6,7'
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_breadth'] = np.log10(df_6['breadth'])
    df_6['log_size_std'] = (df_6['log_size'] - df_6['log_size'].mean()) / df_6['log_size'].std()
    df_6['log_breadth_std'] = (df_6['log_breadth'] - df_6['log_breadth'].mean()) / df_6['log_breadth'].std()
    df_6['platform'] = df_6['platform'].astype('category')
    (y_2, X_3) = patsy.dmatrices('log_breadth ~ log_size * platform', data=df_6, return_type='dataframe')
    y_data = y_2.values.ravel()
    X_mu = X_3[['Intercept', 'platform[T.ts]', 'log_size:platform[T.ts]']]
    X_sigma = X_3[['log_size']]
    from sklearn.preprocessing import StandardScaler
    with pm.Model() as robust_fast_model:
        beta = pm.Normal('beta', mu=0, sigma=1, shape=X_3.shape[1])
        mu = pm.math.dot(X_3, beta)
        gamma = pm.Normal('gamma', mu=0, sigma=0.5, shape=1)
        log_sigma = pm.math.dot(X_sigma, gamma)
        sigma = pm.Deterministic('sigma', pm.math.exp(pm.math.clip(log_sigma, -5, 5)))
        nu = pm.Exponential('nu', 1 / 10, shape=1) + 1
        y_obs = pm.StudentT('y_obs', mu=mu, sigma=sigma, nu=nu, observed=y_data)
        trace = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.95, return_inferencedata=True, idata_kwargs={'log_likelihood': True}, nuts_sampler='numpyro')
    return os, patsy, pm, trace


@app.cell
def _(models, trace):
    models["robust_fast_model"] = trace
    return


@app.cell
def _(az, trace):
    az.summary(trace, var_names=["beta",'gamma'], hdi_prob=0.95)
    return


@app.cell
def _(az, trace):
    az.plot_posterior(trace, var_names=["beta", "gamma"], ref_val=0)
    return


@app.cell
def _(df_6, np, os, patsy, pm):
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,5,6,7'
    df_6['log_size'] = np.log10(df_6['size'])
    df_6['log_breadth'] = np.log10(df_6['breadth'])
    df_6['log_size_std'] = (df_6['log_size'] - df_6['log_size'].mean()) / df_6['log_size'].std()
    df_6['log_breadth_std'] = (df_6['log_breadth'] - df_6['log_breadth'].mean()) / df_6['log_breadth'].std()
    df_6['platform'] = df_6['platform'].astype('category')
    (y_3, X_4) = patsy.dmatrices('log_breadth ~ log_size * platform + log_size * bs(alignment_ratio, df=5)', data=df_6, return_type='dataframe')
    y_data_1 = y_3.values.ravel()
    X_sigma_1 = X_4[['log_size']]
    with pm.Model() as robust_fast_model_1:
        beta_1 = pm.Normal('beta', mu=0, sigma=1, shape=X_4.shape[1])
        mu_1 = pm.math.dot(X_4, beta_1)
        gamma_1 = pm.Normal('gamma', mu=0, sigma=0.5, shape=1)
        log_sigma_1 = pm.math.dot(X_sigma_1, gamma_1)
        sigma_1 = pm.Deterministic('sigma', pm.math.exp(pm.math.clip(log_sigma_1, -5, 5)))
        nu_1 = pm.Exponential('nu', 1 / 10, shape=1) + 1
        y_obs_1 = pm.StudentT('y_obs', mu=mu_1, sigma=sigma_1, nu=nu_1, observed=y_data_1)
        trace_1 = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.95, return_inferencedata=True, idata_kwargs={'log_likelihood': True}, nuts_sampler='numpyro')
    return X_4, trace_1


@app.cell
def _(az, trace_1):
    az.plot_posterior(trace_1, var_names=['beta', 'gamma'], ref_val=0)
    return


@app.cell
def _(df_6, dmatrix, pm):
    platform_idx = df_6['platform'].cat.codes.values
    n_platforms = df_6['platform'].nunique()
    with pm.Model() as model_a:
        mu_alpha = pm.Normal('mu_alpha', 0, 1)
        sigma_alpha = pm.Exponential('sigma_alpha', 1)
        alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_platforms)
        mu_beta = pm.Normal('mu_beta', 0, 1)
        sigma_beta = pm.Exponential('sigma_beta', 1)
        beta_2 = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=n_platforms)
        mu_2 = alpha[platform_idx] + beta_2[platform_idx] * df_6['log_size'].values
        gamma_2 = pm.Normal('gamma', mu=0, sigma=0.5)
        log_sigma_2 = gamma_2 * df_6['log_size'].values
        sigma_2 = pm.Deterministic('sigma', pm.math.exp(pm.math.clip(log_sigma_2, -5, 5)))
        nu_2 = pm.Exponential('nu', 1 / 10) + 1
        y_obs_2 = pm.StudentT('y_obs', mu=mu_2, sigma=sigma_2, nu=nu_2, observed=df_6['log_breadth'])
        trace_a = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True, nuts_sampler='numpyro')
    alignment_spline = dmatrix('bs(alignment_ratio, df=5, include_intercept=False)', data=df_6, return_type='dataframe')
    spline_matrix = alignment_spline.values
    n_spline = spline_matrix.shape[1]
    with pm.Model() as model_b:
        mu_alpha = pm.Normal('mu_alpha', 0, 1)
        sigma_alpha = pm.Exponential('sigma_alpha', 1)
        alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_platforms)
        mu_beta = pm.Normal('mu_beta', 0, 1)
        sigma_beta = pm.Exponential('sigma_beta', 1)
        beta_2 = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=n_platforms)
        gamma_2 = pm.Normal('gamma', mu=0, sigma=1, shape=n_spline)
        size_4 = df_6['log_size'].values
        mu_2 = alpha[platform_idx] + beta_2[platform_idx] * size_4 + pm.math.dot(spline_matrix, gamma_2)
        gamma_sigma = pm.Normal('gamma_sigma', mu=0, sigma=0.5)
        log_sigma_2 = gamma_sigma * size_4
        sigma_2 = pm.Deterministic('sigma', pm.math.exp(pm.math.clip(log_sigma_2, -5, 5)))
        nu_2 = pm.Exponential('nu', 1 / 10) + 1
        y_obs_2 = pm.StudentT('y_obs', mu=mu_2, sigma=sigma_2, nu=nu_2, observed=df_6['log_breadth'])
        trace_b = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True, nuts_sampler='numpyro')
    return trace_a, trace_b


@app.cell
def _(az, trace_a):
    az.summary(trace_a, var_names=["alpha", "beta", ], hdi_prob=0.95)
    return


@app.cell
def _(az, trace_b):
    az.summary(trace_b, var_names=["alpha", "beta", ], hdi_prob=0.95)
    return


@app.cell
def _(az, trace_a, trace_b):
    az.compare({"model_a": trace_a, "model_b": trace_b},)
    return


@app.cell
def _(az, trace_a, trace_b):
    az.plot_forest(trace_a, var_names=["beta"], combined=True, r_hat=True)
    az.plot_forest(trace_b, var_names=["beta"], combined=True, r_hat=True)
    return


@app.cell
def _(X_4, df_6, plt, spline_start):
    for i_9 in range(5):
        plt.plot(df_6['alignment_ratio'], X_4.iloc[:, spline_start + i_9], label=f'spline_{i_9}')
    plt.legend()
    plt.title('B-spline Basis over Alignment Ratio')
    plt.xlabel('alignment_ratio')
    plt.ylabel('Basis value')
    plt.show()
    return


@app.cell
def _(az, models):
    comparison = az.compare(models)
    az.plot_compare(comparison, figsize=(8, 4))
    comparison
    return


@app.cell
def _(bsky_df_5, np, pd, plt, sm, ts_df_5):
    from matplotlib.lines import Line2D
    import itertools
    colors_7 = {'bsky': '#007FFF', 'ts': '#FFD700'}
    df_7 = pd.concat([bsky_df_5, ts_df_5], ignore_index=True)
    df_7['topic_label'] = df_7['topic_label'].replace({'MAGA and Pro-Trump Hashtags and Advocacy': 'Pro-Trump and MAGA Advocacy'})
    df_7 = df_7[df_7['topic_label'] != 'Criticism of Trump and Support for Democratic Policies']
    df_7['log_size'] = np.log10(df_7['size'])
    df_7['log_breadth'] = np.log10(df_7['breadth'])
    line_styles = ['-', '--', ':', '-.']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h', 'H', '+']
    style_cycle = itertools.cycle(zip(line_styles * len(markers), markers))
    topic_style_map = {topic: next(style_cycle) for topic in df_7['topic_label'].unique()}

    def robust_fit_4(x, y):
        x = sm.add_constant(x)
        model = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
        x_sorted = np.sort(x[:, 1])
        x_pred = sm.add_constant(x_sorted)
        y_pred = model.predict(x_pred)
        return (x_sorted, y_pred, model)
    plt.figure(figsize=(14, 9), dpi=300)
    for platform_24 in ['bsky', 'ts']:
        for topic in df_7['topic_label'].unique():
            subset = df_7[(df_7['platform'] == platform_24) & (df_7['topic_label'] == topic)]
            if len(subset) < 5:
                continue
            x_vals_3 = subset['log_size'].values
            y_vals_3 = subset['log_breadth'].values
            (x_fit_2, y_fit_2, model_10) = robust_fit_4(x_vals_3, y_vals_3)
            slope_3 = model_10.params[1]
            stderr_3 = model_10.bse[1]
            print(f'{platform_24.upper()} - {topic}: β = {slope_3:.4f}; Standard Error: {stderr_3:.4f}')
            (linestyle_1, marker_1) = topic_style_map[topic]
            plt.plot(x_fit_2, y_fit_2, color=colors_7[platform_24], linestyle=linestyle_1, linewidth=2, label=f'{platform_24.upper()} - {topic}')
            plt.scatter(x_vals_3, y_vals_3, alpha=0.15, s=10, marker=marker_1, color=colors_7[platform_24])
            plt.text(x_fit_2[-1], y_fit_2[-1], f'β={slope_3:.2f}', fontsize=9, ha='left', va='center', color=colors_7[platform_24], bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    topic_legend = [Line2D([0], [0], linestyle=linestyle, marker=marker, color='gray', label=topic) for (topic, (linestyle, marker)) in topic_style_map.items()]
    platform_legend = [Line2D([0], [0], linestyle='-', color=color, label=name.upper()) for (name, color) in colors_7.items()]
    plt.legend(handles=platform_legend + topic_legend, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.xlabel('Log(Size)')
    plt.ylabel('Log(Breadth)')
    plt.title('Robust Regression of Breadth by Platform and Topic')
    plt.grid(alpha=0.3)
    plt.show()
    return Line2D, itertools


@app.cell
def _(bsky_df_5, f_1, np, pd, plt, sm, ts_df_5):
    colors_8 = {'bsky_scatter': '#007FFF', 'bsky_rlm': '#0056A3', 'ts_scatter': '#FFD700', 'ts_rlm': '#C49A00'}
    df_8 = pd.concat([bsky_df_5, ts_df_5], ignore_index=True)
    df_bsky_2 = df_8[df_8['platform'] == 'bsky'].copy()
    df_ts_2 = df_8[df_8['platform'] == 'ts'].copy()
    df_bsky_2['log_size'] = np.log10(df_bsky_2['size'])
    df_bsky_2['log_breadth'] = np.log10(df_bsky_2['breadth'])
    df_ts_2['log_size'] = np.log10(df_ts_2['size'])
    df_ts_2['log_breadth'] = np.log10(df_ts_2['breadth'])

    def robust_fit_5(x, y):
        """Fits a robust regression model and returns sorted predictions."""
        x = sm.add_constant(x)
        model = sm.RLM(y, x, M=sm.robust.norms.HuberT())
        results = model.fit()
        x_sorted = np.sort(x[:, 1])
        x_pred = sm.add_constant(x_sorted)
        y_pred = results.predict(x_pred)
        print(results.summary())
        return (x_sorted, y_pred, results)
    (bsky_x_1, bsky_y_1, model_bsky_1) = robust_fit_5(df_bsky_2['log_size'].values, df_bsky_2['log_breadth'].values)
    (ts_x_1, ts_y_1, model_ts_1) = robust_fit_5(df_ts_2['log_size'].values, df_ts_2['log_breadth'].values)
    bsky_slope_1 = model_bsky_1.params[1]
    ts_slope_1 = model_ts_1.params[1]

    def chow_test_1(x1, y1, x2, y2):
        """Performs the Chow test to compare regression slopes between two datasets."""
        (x1_const, x2_const) = (sm.add_constant(x1), sm.add_constant(x2))
        model1 = sm.RLM(y1, x1_const, M=sm.robust.norms.HuberT()).fit()
        model2 = sm.RLM(y2, x2_const, M=sm.robust.norms.HuberT()).fit()
        x_combined = np.concatenate([x1, x2])
        y_combined = np.concatenate([y1, y2])
        x_combined_const = sm.add_constant(x_combined)
        model_combined = sm.RLM(y_combined, x_combined_const, M=sm.robust.norms.HuberT()).fit()
        SSR_combined = np.sum(model_combined.resid ** 2)
        SSR1 = np.sum(model1.resid ** 2)
        SSR2 = np.sum(model2.resid ** 2)
        (n1, n2) = (len(y1), len(y2))
        k = 2
        chow_stat = (SSR_combined - (SSR1 + SSR2)) / k / ((SSR1 + SSR2) / (n1 + n2 - 2 * k))
        p_value = 1 - f_1.cdf(chow_stat, k, n1 + n2 - 2 * k)
        return (chow_stat, p_value)
    (chow_stat_1, p_value_15) = chow_test_1(df_bsky_2['log_size'].values, df_bsky_2['log_breadth'].values, df_ts_2['log_size'].values, df_ts_2['log_breadth'].values)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(df_bsky_2['log_size'], df_bsky_2['log_breadth'], alpha=0.3, color=colors_8['bsky_scatter'], label='BlueSky Data')
    plt.scatter(df_ts_2['log_size'], df_ts_2['log_breadth'], alpha=0.3, color=colors_8['ts_scatter'], label='TruthSocial Data')
    plt.plot(bsky_x_1, bsky_y_1, color=colors_8['bsky_rlm'], linewidth=2, label='BlueSky Robust Fit')
    plt.plot(ts_x_1, ts_y_1, color=colors_8['ts_rlm'], linewidth=2, label='TruthSocial Robust Fit')
    plt.xlabel('Size)')
    plt.ylabel('Breadth')
    plt.title('Robust Regression of Breadth Across Platforms in Reply Network')
    plt.text(min(df_bsky_2['log_size']), max(df_bsky_2['log_breadth']), f'BlueSky Slope: {bsky_slope_1:.4f}\nTruthSocial Slope: {ts_slope_1:.4f}', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(bsky_df_5, f_1, np, pd, plt, sm, ts_df_5):
    colors_9 = {'bsky_scatter': '#007FFF', 'bsky_rlm': '#0056A3', 'ts_scatter': '#FFD700', 'ts_rlm': '#C49A00'}
    df_9 = pd.concat([bsky_df_5, ts_df_5.loc[ts_df_5['outlier'] != True, :]], ignore_index=True)
    df_bsky_3 = df_9[df_9['platform'] == 'bsky'].copy()
    df_ts_3 = df_9[df_9['platform'] == 'ts'].copy()
    df_bsky_3['log_size'] = np.log10(df_bsky_3['size'])
    df_bsky_3['log_breadth'] = np.log10(df_bsky_3['breadth'])
    df_ts_3['log_size'] = np.log10(df_ts_3['size'])
    df_ts_3['log_breadth'] = np.log10(df_ts_3['breadth'])

    def robust_fit_6(x, y):
        """Fits a robust regression model and returns sorted predictions."""
        x = sm.add_constant(x)
        model = sm.RLM(y, x, M=sm.robust.norms.HuberT())
        results = model.fit()
        x_sorted = np.sort(x[:, 1])
        x_pred = sm.add_constant(x_sorted)
        y_pred = results.predict(x_pred)
        print(results.summary())
        return (x_sorted, y_pred, results)
    (bsky_x_2, bsky_y_2, model_bsky_2) = robust_fit_6(df_bsky_3['log_size'].values, df_bsky_3['log_breadth'].values)
    (ts_x_2, ts_y_2, model_ts_2) = robust_fit_6(df_ts_3['log_size'].values, df_ts_3['log_breadth'].values)
    bsky_slope_2 = model_bsky_2.params[1]
    ts_slope_2 = model_ts_2.params[1]

    def chow_test_2(x1, y1, x2, y2):
        """Performs the Chow test to compare regression slopes between two datasets."""
        (x1_const, x2_const) = (sm.add_constant(x1), sm.add_constant(x2))
        model1 = sm.RLM(y1, x1_const, M=sm.robust.norms.HuberT()).fit()
        model2 = sm.RLM(y2, x2_const, M=sm.robust.norms.HuberT()).fit()
        x_combined = np.concatenate([x1, x2])
        y_combined = np.concatenate([y1, y2])
        x_combined_const = sm.add_constant(x_combined)
        model_combined = sm.RLM(y_combined, x_combined_const, M=sm.robust.norms.HuberT()).fit()
        SSR_combined = np.sum(model_combined.resid ** 2)
        SSR1 = np.sum(model1.resid ** 2)
        SSR2 = np.sum(model2.resid ** 2)
        (n1, n2) = (len(y1), len(y2))
        k = 2
        chow_stat = (SSR_combined - (SSR1 + SSR2)) / k / ((SSR1 + SSR2) / (n1 + n2 - 2 * k))
        p_value = 1 - f_1.cdf(chow_stat, k, n1 + n2 - 2 * k)
        return (chow_stat, p_value)
    (chow_stat_2, p_value_16) = chow_test_2(df_bsky_3['log_size'].values, df_bsky_3['log_breadth'].values, df_ts_3['log_size'].values, df_ts_3['log_breadth'].values)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(df_bsky_3['log_size'], df_bsky_3['log_breadth'], alpha=0.3, color=colors_9['bsky_scatter'], label='BlueSky Data')
    plt.scatter(df_ts_3['log_size'], df_ts_3['log_breadth'], alpha=0.3, color=colors_9['ts_scatter'], label='TruthSocial Data')
    plt.plot(bsky_x_2, bsky_y_2, color=colors_9['bsky_rlm'], linewidth=2, label='BlueSky Robust Fit')
    plt.plot(ts_x_2, ts_y_2, color=colors_9['ts_rlm'], linewidth=2, label='TruthSocial Robust Fit')
    plt.xlabel('Size)')
    plt.ylabel('Breadth')
    plt.title('Robust Regression of Breadth Across Platforms in Reply Network')
    plt.text(min(df_bsky_3['log_size']), max(df_bsky_3['log_breadth']), f'BlueSky Slope: {bsky_slope_2:.4f}\nTruthSocial Slope: {ts_slope_2:.4f}', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return (df_9,)


@app.cell
def _(df_9, np, plt):
    colors_10 = {'bsky_scatter': '#007FFF', 'bsky_kernel': '#0056A3', 'ts_scatter': '#FFD700', 'ts_kernel': '#C49A00'}
    df_bsky_4 = df_9[df_9['platform'] == 'bsky'].copy()
    df_ts_4 = df_9[df_9['platform'] == 'ts'].copy()
    df_bsky_4['log_size'] = np.log10(df_bsky_4['size'])
    df_bsky_4['log_breadth'] = np.log10(df_bsky_4['breadth'])
    df_ts_4['log_size'] = np.log10(df_ts_4['size'])
    df_ts_4['log_breadth'] = np.log10(df_ts_4['breadth'])

    def kernel_regression(x_train, y_train, x_pred, bandwidth=0.3):
        """Performs Nadaraya-Watson Kernel Regression using Gaussian kernel."""
        y_pred = np.zeros_like(x_pred)
        for (i, x) in enumerate(x_pred):
            weights = np.exp(-(x_train - x) ** 2 / (2 * bandwidth ** 2))
            weights = weights / weights.sum()
            y_pred[i] = np.sum(weights * y_train)
        return y_pred

    def bootstrap_ci_kernel(x_train, y_train, x_pred, bandwidth=0.3, n_bootstrap=1000, alpha=0.05):
        """Computes bootstrap confidence intervals for kernel regression."""
        bootstrap_preds = np.zeros((n_bootstrap, len(x_pred)))
        n = len(y_train)
        for i in range(n_bootstrap):
            sample_idx = np.random.choice(n, n, replace=True)
            (x_sample, y_sample) = (x_train[sample_idx], y_train[sample_idx])
            bootstrap_preds[i, :] = kernel_regression(x_sample, y_sample, x_pred, bandwidth)
        lower_ci = np.percentile(bootstrap_preds, 100 * alpha / 2, axis=0)
        upper_ci = np.percentile(bootstrap_preds, 100 * (1 - alpha / 2), axis=0)
        return (lower_ci, upper_ci)
    x_pred_bsky = np.linspace(df_bsky_4['log_size'].min(), df_bsky_4['log_size'].max(), 200)
    x_pred_ts = np.linspace(df_ts_4['log_size'].min(), df_ts_4['log_size'].max(), 200)
    bsky_y_pred = kernel_regression(df_bsky_4['log_size'].values, df_bsky_4['log_breadth'].values, x_pred_bsky)
    ts_y_pred = kernel_regression(df_ts_4['log_size'].values, df_ts_4['log_breadth'].values, x_pred_ts)
    (bsky_lower_ci, bsky_upper_ci) = bootstrap_ci_kernel(df_bsky_4['log_size'].values, df_bsky_4['log_breadth'].values, x_pred_bsky)
    (ts_lower_ci, ts_upper_ci) = bootstrap_ci_kernel(df_ts_4['log_size'].values, df_ts_4['log_breadth'].values, x_pred_ts)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(df_bsky_4['log_size'], df_bsky_4['log_breadth'], alpha=0.1, color=colors_10['bsky_scatter'], label='BlueSky Data')
    plt.scatter(df_ts_4['log_size'], df_ts_4['log_breadth'], alpha=0.1, color=colors_10['ts_scatter'], label='TruthSocial Data')
    plt.plot(x_pred_bsky, bsky_y_pred, color=colors_10['bsky_kernel'], linewidth=2, label='BlueSky Kernel Fit')
    plt.plot(x_pred_ts, ts_y_pred, color=colors_10['ts_kernel'], linewidth=2, label='TruthSocial Kernel Fit')
    plt.fill_between(x_pred_bsky, bsky_lower_ci, bsky_upper_ci, color=colors_10['bsky_kernel'], alpha=0.2, label='BlueSky 95% CI')
    plt.fill_between(x_pred_ts, ts_lower_ci, ts_upper_ci, color=colors_10['ts_kernel'], alpha=0.2, label='TruthSocial 95% CI')
    plt.xlabel('Size')
    plt.ylabel('Breadth')
    plt.title('Kernel Regression with Confidence Intervals')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(bsky_df_5, np, pd, plt, ts_df_5):
    colors_11 = {'bsky_scatter': '#007FFF', 'bsky_kernel': '#0056A3', 'ts_scatter': '#FFD700', 'ts_kernel': '#C49A00'}
    df_10 = pd.concat([bsky_df_5, ts_df_5], ignore_index=True)
    df_bsky_5 = df_10[df_10['platform'] == 'bsky'].copy()
    df_ts_5 = df_10[df_10['platform'] == 'ts'].copy()
    df_bsky_5['log_size'] = np.log10(df_bsky_5['size'])
    df_bsky_5['log_max_depth'] = np.log10(df_bsky_5['max_depth'] + 1)
    df_ts_5['log_size'] = np.log10(df_ts_5['size'])
    df_ts_5['log_max_depth'] = np.log10(df_ts_5['max_depth'] + 1)

    def kernel_regression_1(x_train, y_train, x_pred, bandwidth=0.3):
        """Performs Nadaraya-Watson Kernel Regression using Gaussian kernel."""
        y_pred = np.zeros_like(x_pred)
        for (i, x) in enumerate(x_pred):
            weights = np.exp(-(x_train - x) ** 2 / (2 * bandwidth ** 2))
            weights = weights / weights.sum()
            y_pred[i] = np.sum(weights * y_train)
        return y_pred
    x_pred_bsky_1 = np.linspace(df_bsky_5['log_size'].min(), df_bsky_5['log_size'].max(), 200)
    x_pred_ts_1 = np.linspace(df_ts_5['log_size'].min(), df_ts_5['log_size'].max(), 200)
    bsky_y_pred_1 = kernel_regression_1(df_bsky_5['log_size'].values, df_bsky_5['log_max_depth'].values, x_pred_bsky_1)
    ts_y_pred_1 = kernel_regression_1(df_ts_5['log_size'].values, df_ts_5['log_max_depth'].values, x_pred_ts_1)
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(x_pred_bsky_1, bsky_y_pred_1, color=colors_11['bsky_kernel'], linewidth=2, label='BlueSky Kernel Fit')
    plt.plot(x_pred_ts_1, ts_y_pred_1, color=colors_11['ts_kernel'], linewidth=2, label='TruthSocial Kernel Fit')
    plt.xlabel('Size')
    plt.ylabel('max_depth')
    plt.title('Kernel Regression of max_depth Across Platforms')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(bsky_df_5, f_1, np, pd, plt, sm, ts_df_5):
    colors_12 = {'bsky_scatter': '#007FFF', 'bsky_rlm': '#0056A3', 'ts_scatter': '#FFD700', 'ts_rlm': '#C49A00'}
    df_11 = pd.concat([bsky_df_5, ts_df_5], ignore_index=True)
    df_bsky_6 = df_11[df_11['platform'] == 'bsky'].copy()
    df_ts_6 = df_11[df_11['platform'] == 'ts'].copy()
    df_bsky_6['log_size'] = np.log10(df_bsky_6['size'])
    df_bsky_6['log_max_depth'] = np.log10(df_bsky_6['max_depth'] + 1)
    df_ts_6['log_size'] = np.log10(df_ts_6['size'])
    df_ts_6['log_max_depth'] = np.log10(df_ts_6['max_depth'] + 1)

    def robust_fit_7(x, y):
        """Fits a robust regression model and returns sorted predictions."""
        x_const = sm.add_constant(x)
        model = sm.RLM(y, x_const, M=sm.robust.norms.HuberT())
        results = model.fit()
        x_sorted = np.sort(x)
        x_pred = sm.add_constant(x_sorted)
        y_pred = results.predict(x_pred)
        return (x_sorted, y_pred, results)
    (bsky_x_3, bsky_y_3, model_bsky_3) = robust_fit_7(df_bsky_6['log_size'].values, df_bsky_6['log_max_depth'].values)
    (ts_x_3, ts_y_3, model_ts_3) = robust_fit_7(df_ts_6['log_size'].values, df_ts_6['log_max_depth'].values)
    bsky_slope_3 = model_bsky_3.params[1]
    ts_slope_3 = model_ts_3.params[1]

    def chow_test_3(x1, y1, x2, y2):
        """Performs the Chow test to compare regression slopes between two datasets."""
        (x1_const, x2_const) = (sm.add_constant(x1), sm.add_constant(x2))
        model1 = sm.RLM(y1, x1_const, M=sm.robust.norms.HuberT()).fit()
        model2 = sm.RLM(y2, x2_const, M=sm.robust.norms.HuberT()).fit()
        x_combined = np.concatenate([x1, x2])
        y_combined = np.concatenate([y1, y2])
        x_combined_const = sm.add_constant(x_combined)
        model_combined = sm.RLM(y_combined, x_combined_const, M=sm.robust.norms.HuberT()).fit()
        SSR_combined = np.sum(model_combined.resid ** 2)
        SSR1 = np.sum(model1.resid ** 2)
        SSR2 = np.sum(model2.resid ** 2)
        (n1, n2) = (len(y1), len(y2))
        k = 2
        chow_stat = (SSR_combined - (SSR1 + SSR2)) / k / ((SSR1 + SSR2) / (n1 + n2 - 2 * k))
        p_value = 1 - f_1.cdf(chow_stat, k, n1 + n2 - 2 * k)
        return (chow_stat, p_value)
    (chow_stat_3, p_value_17) = chow_test_3(df_bsky_6['log_size'].values, df_bsky_6['log_max_depth'].values, df_ts_6['log_size'].values, df_ts_6['log_max_depth'].values)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(df_bsky_6['log_size'], df_bsky_6['log_max_depth'], alpha=0.3, color=colors_12['bsky_scatter'], label='BlueSky Data')
    plt.scatter(df_ts_6['log_size'], df_ts_6['log_max_depth'], alpha=0.3, color=colors_12['ts_scatter'], label='TruthSocial Data')
    plt.plot(bsky_x_3, bsky_y_3, color=colors_12['bsky_rlm'], linewidth=2, label='BlueSky Robust Fit')
    plt.plot(ts_x_3, ts_y_3, color=colors_12['ts_rlm'], linewidth=2, label='TruthSocial Robust Fit')
    plt.xlabel('Size')
    plt.ylabel('Depth')
    plt.title('Robust Regression of Depth Across Platforms in Reply Network')
    plt.text(min(df_bsky_6['log_size']), max(df_bsky_6['log_max_depth']), f'BlueSky Slope: {bsky_slope_3:.4f}\nTruthSocial Slope: {ts_slope_3:.4f}', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(Line2D, bsky_df_5, np, pd, plt, sm, ts_df_5):
    colors_13 = {'bsky': '#007FFF', 'ts': '#FFD700'}
    low_thresh_1 = 0.4
    high_thresh_1 = 0.6
    df_12 = pd.concat([bsky_df_5, ts_df_5], ignore_index=True)
    df_12['alignment_category'] = pd.cut(df_12['alignment_ratio'], bins=[-np.inf, low_thresh_1, high_thresh_1, np.inf], labels=['Low', 'Medium', 'High'])
    df_12['log_size'] = np.log10(df_12['size'])
    df_12['log_max_depth'] = np.log10(df_12['max_depth'] + 1)

    def robust_fit_8(x, y):
        x = sm.add_constant(x)
        model = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
        x_sorted = np.sort(x[:, 1])
        x_pred = sm.add_constant(x_sorted)
        y_pred = model.predict(x_pred)
        return (x_sorted, y_pred, model)
    plt.figure(figsize=(12, 8), dpi=300)
    for platform_25 in ['bsky', 'ts']:
        for category_1 in ['Low', 'Medium', 'High']:
            subset_1 = df_12[(df_12['platform'] == platform_25) & (df_12['alignment_category'] == category_1)]
            if len(subset_1) < 5:
                continue
            x_vals_4 = subset_1['log_size'].values
            y_vals_4 = subset_1['log_max_depth'].values
            (x_fit_3, y_fit_3, model_11) = robust_fit_8(x_vals_4, y_vals_4)
            plt.scatter(x_vals_4, y_vals_4, alpha=0.25, label=f'{platform_25.upper()} - {category_1}', color=colors_13[platform_25])
            linestyle_2 = '-' if category_1 == 'Low' else '--' if category_1 == 'Medium' else ':'
            plt.plot(x_fit_3, y_fit_3, linewidth=2, linestyle=linestyle_2, color=colors_13[platform_25])
            slope_4 = model_11.params[1]
            label_x_1 = x_fit_3[-1]
            label_y_1 = y_fit_3[-1]
            plt.text(label_x_1, label_y_1, f'β={slope_4:.2f}', fontsize=9, ha='left', va='center', color=colors_13[platform_25], bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    alignment_legend = [Line2D([0], [0], color='gray', linestyle='-', label='Low Alignment'), Line2D([0], [0], color='gray', linestyle='--', label='Medium Alignment'), Line2D([0], [0], color='gray', linestyle=':', label='High Alignment')]
    (handles_11, labels_13) = plt.gca().get_legend_handles_labels()
    plt.legend(handles_11 + alignment_legend, labels_13 + ['Low Alignment', 'Medium Alignment', 'High Alignment'])
    plt.xlabel('Log(Size)')
    plt.ylabel('Log(Max Depth + 1)')
    plt.title('Robust Regression of Max Depth by Platform and Alignment Category')
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(bsky_df_5, f_1, np, pd, plt, sm, ts_df_5):
    colors_14 = {'bsky_scatter': '#007FFF', 'bsky_rlm': '#0056A3', 'ts_scatter': '#FFD700', 'ts_rlm': '#C49A00'}
    df_13 = pd.concat([bsky_df_5, ts_df_5.loc[ts_df_5['outlier'] != True, :]], ignore_index=True)
    df_bsky_7 = df_13[df_13['platform'] == 'bsky'].copy()
    df_ts_7 = df_13[df_13['platform'] == 'ts'].copy()
    df_bsky_7['log_size'] = np.log10(df_bsky_7['size'])
    df_bsky_7['log_max_depth'] = np.log10(df_bsky_7['max_depth'] + 1)
    df_ts_7['log_size'] = np.log10(df_ts_7['size'])
    df_ts_7['log_max_depth'] = np.log10(df_ts_7['max_depth'] + 1)

    def robust_fit_9(x, y):
        """Fits a robust regression model and returns sorted predictions."""
        x_const = sm.add_constant(x)
        model = sm.RLM(y, x_const, M=sm.robust.norms.HuberT())
        results = model.fit()
        x_sorted = np.sort(x)
        x_pred = sm.add_constant(x_sorted)
        y_pred = results.predict(x_pred)
        return (x_sorted, y_pred, results)
    (bsky_x_4, bsky_y_4, model_bsky_4) = robust_fit_9(df_bsky_7['log_size'].values, df_bsky_7['log_max_depth'].values)
    (ts_x_4, ts_y_4, model_ts_4) = robust_fit_9(df_ts_7['log_size'].values, df_ts_7['log_max_depth'].values)
    bsky_slope_4 = model_bsky_4.params[1]
    ts_slope_4 = model_ts_4.params[1]

    def chow_test_4(x1, y1, x2, y2):
        """Performs the Chow test to compare regression slopes between two datasets."""
        (x1_const, x2_const) = (sm.add_constant(x1), sm.add_constant(x2))
        model1 = sm.RLM(y1, x1_const, M=sm.robust.norms.HuberT()).fit()
        model2 = sm.RLM(y2, x2_const, M=sm.robust.norms.HuberT()).fit()
        x_combined = np.concatenate([x1, x2])
        y_combined = np.concatenate([y1, y2])
        x_combined_const = sm.add_constant(x_combined)
        model_combined = sm.RLM(y_combined, x_combined_const, M=sm.robust.norms.HuberT()).fit()
        SSR_combined = np.sum(model_combined.resid ** 2)
        SSR1 = np.sum(model1.resid ** 2)
        SSR2 = np.sum(model2.resid ** 2)
        (n1, n2) = (len(y1), len(y2))
        k = 2
        chow_stat = (SSR_combined - (SSR1 + SSR2)) / k / ((SSR1 + SSR2) / (n1 + n2 - 2 * k))
        p_value = 1 - f_1.cdf(chow_stat, k, n1 + n2 - 2 * k)
        return (chow_stat, p_value)
    (chow_stat_4, p_value_18) = chow_test_4(df_bsky_7['log_size'].values, df_bsky_7['log_max_depth'].values, df_ts_7['log_size'].values, df_ts_7['log_max_depth'].values)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(df_bsky_7['log_size'], df_bsky_7['log_max_depth'], alpha=0.3, color=colors_14['bsky_scatter'], label='BlueSky Data')
    plt.scatter(df_ts_7['log_size'], df_ts_7['log_max_depth'], alpha=0.3, color=colors_14['ts_scatter'], label='TruthSocial Data')
    plt.plot(bsky_x_4, bsky_y_4, color=colors_14['bsky_rlm'], linewidth=2, label='BlueSky Robust Fit')
    plt.plot(ts_x_4, ts_y_4, color=colors_14['ts_rlm'], linewidth=2, label='TruthSocial Robust Fit')
    plt.xlabel('Size')
    plt.ylabel('Depth')
    plt.title('Robust Regression of Depth Across Platforms in Reply Network')
    plt.text(min(df_bsky_7['log_size']), max(df_bsky_7['log_max_depth']), f'BlueSky Slope: {bsky_slope_4:.4f}\nTruthSocial Slope: {ts_slope_4:.4f}', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(Line2D, bsky_df_5, itertools, np, pd, plt, sm, ts_df_5):
    colors_15 = {'bsky': '#007FFF', 'ts': '#FFD700'}
    df_14 = pd.concat([bsky_df_5, ts_df_5], ignore_index=True)
    df_14['log_size'] = np.log10(df_14['size'])
    df_14['log_max_depth'] = np.log10(df_14['max_depth'] + 1)
    line_styles_1 = ['-', '--', ':', '-.']
    markers_1 = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h', 'H', '+']
    style_cycle_1 = itertools.cycle(zip(line_styles_1 * len(markers_1), markers_1))
    topic_style_map_1 = {topic: next(style_cycle_1) for topic in df_14['topic_label'].unique()}

    def robust_fit_10(x, y):
        x = sm.add_constant(x)
        model = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
        x_sorted = np.sort(x[:, 1])
        x_pred = sm.add_constant(x_sorted)
        y_pred = model.predict(x_pred)
        return (x_sorted, y_pred, model)
    plt.figure(figsize=(14, 9), dpi=300)
    for platform_26 in ['bsky', 'ts']:
        for topic_1 in df_14['topic_label'].unique():
            subset_2 = df_14[(df_14['platform'] == platform_26) & (df_14['topic_label'] == topic_1)]
            if len(subset_2) < 5:
                continue
            x_vals_5 = subset_2['log_size'].values
            y_vals_5 = subset_2['log_max_depth'].values
            (x_fit_4, y_fit_4, model_12) = robust_fit_10(x_vals_5, y_vals_5)
            slope_5 = model_12.params[1]
            print(f'{platform_26.upper()} - {topic_1}: β = {slope_5:.4f}')
            (linestyle_3, marker_2) = topic_style_map_1[topic_1]
            plt.plot(x_fit_4, y_fit_4, color=colors_15[platform_26], linestyle=linestyle_3, linewidth=2, label=f'{platform_26.upper()} - {topic_1}')
            plt.scatter(x_vals_5, y_vals_5, alpha=0.15, s=10, marker=marker_2, color=colors_15[platform_26])
            plt.text(x_fit_4[-1], y_fit_4[-1], f'β={slope_5:.2f}', fontsize=9, ha='left', va='center', color=colors_15[platform_26], bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    topic_legend_1 = [Line2D([0], [0], linestyle=linestyle, marker=marker, color='gray', label=topic) for (topic, (linestyle, marker)) in topic_style_map_1.items()]
    platform_legend_1 = [Line2D([0], [0], linestyle='-', color=color, label=name.upper()) for (name, color) in colors_15.items()]
    plt.legend(handles=platform_legend_1 + topic_legend_1, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.xlabel('Log(Size)')
    plt.ylabel('Log(Max Depth + 1)')
    plt.title('Robust Regression of Max Depth by Platform and Topic')
    plt.grid(alpha=0.3)
    plt.show()
    return (df_14,)


@app.cell
def _(df_14, np, plt):
    import matplotlib.lines as mlines
    colors_16 = {'left': ('#436685', '#2A4765'), 'center': ('#bbcd78', '#8FA34F'), 'right': ('#8a2520', '#5E1815')}
    df_bsky_8 = df_14[df_14['platform'] == 'bsky'].copy()
    df_ts_8 = df_14[df_14['platform'] == 'ts'].copy()
    for df_platform_1 in [df_bsky_8, df_ts_8]:
        df_platform_1['log_size'] = np.log10(df_platform_1['size'])
        df_platform_1['log_max_depth'] = np.log10(df_platform_1['max_depth'] + 1)

    def kernel_regression_2(x_train, y_train, x_pred, bandwidth=0.5):
        """Performs Nadaraya-Watson Kernel Regression using Gaussian kernel."""
        y_pred = np.zeros_like(x_pred)
        for (i, x) in enumerate(x_pred):
            weights = np.exp(-(x_train - x) ** 2 / (2 * bandwidth ** 2))
            weights = weights / weights.sum()
            y_pred[i] = np.sum(weights * y_train)
        return y_pred
    (fig_33, axes_33) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    def plot_kernel_by_partisanship(df_platform, ax):
        for partisanship in ['left', 'center', 'right']:
            df_subset = df_platform[df_platform['partisanship'] == partisanship]
            if df_subset.empty:
                continue
            x_pred = np.linspace(df_subset['log_size'].min(), df_subset['log_size'].max(), 200)
            y_pred = kernel_regression_2(df_subset['log_size'].values, df_subset['log_max_depth'].values, x_pred)
            (scatter_color, line_color) = colors_16[partisanship]
            ax.scatter(df_subset['log_size'], df_subset['log_max_depth'], alpha=0.2, color=scatter_color)
            ax.plot(x_pred, y_pred, color=line_color, linewidth=2)
    plot_kernel_by_partisanship(df_bsky_8, axes_33[0])
    axes_33[0].set_title('BlueSky: Kernel Smoothed Depth by Partisanship')
    plot_kernel_by_partisanship(df_ts_8, axes_33[1])
    axes_33[1].set_title('TruthSocial: Kernel Smoothed Depth by Partisanship')
    for ax_7 in axes_33:
        ax_7.set_xlabel('Log(Size)')
        ax_7.set_ylabel('Log(Depth)')
        ax_7.grid(alpha=0.3)
    line_legend = [mlines.Line2D([], [], color=colors_16['left'][1], linewidth=2, label='Left'), mlines.Line2D([], [], color=colors_16['center'][1], linewidth=2, label='Center'), mlines.Line2D([], [], color=colors_16['right'][1], linewidth=2, label='Right')]
    fig_33.legend(handles=line_legend, title='Kernel Regression Line Color Legend', loc='upper center', bbox_to_anchor=(0.17, 0.85), ncol=3)
    plt.tight_layout()
    plt.show()
    return (mlines,)


@app.cell
def _(df_14, mlines, np, plt, sm):
    from scipy.stats import norm
    colors_17 = {'left': ('#436685', '#2A4765'), 'center': ('#bbcd78', '#8FA34F'), 'right': ('#8a2520', '#5E1815')}
    df_bsky_9 = df_14[df_14['platform'] == 'bsky'].copy()
    df_ts_9 = df_14[df_14['platform'] == 'ts'].copy()
    for df_platform_2 in [df_bsky_9, df_ts_9]:
        df_platform_2['log_size'] = np.log10(df_platform_2['size'])
        df_platform_2['log_max_depth'] = np.log10(df_platform_2['max_depth'] + 1)

    def robust_fit_11(x, y):
        """Fits a robust regression model and returns sorted predictions and model."""
        x_const = sm.add_constant(x)
        model = sm.RLM(y, x_const, M=sm.robust.norms.HuberT()).fit()
        x_sorted = np.sort(x)
        x_pred = sm.add_constant(x_sorted)
        y_pred = model.predict(x_pred)
        return (x_sorted, y_pred, model)
    (fig_34, axes_34) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    def plot_robust_by_partisanship(df_platform, ax, platform_name):
        slopes = {}
        for partisanship in ['left', 'center', 'right']:
            df_subset = df_platform[df_platform['partisanship'] == partisanship]
            if df_subset.empty:
                continue
            (x_sorted, y_pred, model) = robust_fit_11(df_subset['log_size'].values, df_subset['log_max_depth'].values)
            (slope, se) = (model.params[1], model.bse[1])
            slopes[partisanship] = (slope, se)
            (scatter_color, line_color) = colors_17[partisanship]
            ax.scatter(df_subset['log_size'], df_subset['log_max_depth'], alpha=0.2, color=scatter_color)
            ax.plot(x_sorted, y_pred, color=line_color, linewidth=2, label=f'{partisanship.capitalize()} Fit')
        if 'left' in slopes and 'right' in slopes:
            (left_slope, left_se) = slopes['left']
            (right_slope, right_se) = slopes['right']
            z_score = (left_slope - right_slope) / np.sqrt(left_se ** 2 + right_se ** 2)
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            ax.text(min(df_platform['log_size']), max(df_platform['log_max_depth']), f'Paternoster Z-test: Z={z_score:.2f}, p={p_value:.4f}\nLeft Slope: {left_slope:.4f}\nRight Slope: {right_slope:.4f}', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plot_robust_by_partisanship(df_bsky_9, axes_34[0], 'BlueSky')
    axes_34[0].set_title('BlueSky: Robust Regression Depth by Partisanship')
    plot_robust_by_partisanship(df_ts_9, axes_34[1], 'TruthSocial')
    axes_34[1].set_title('TruthSocial: Robust Regression Depth by Partisanship')
    for ax_8 in axes_34:
        ax_8.set_xlabel('Log(Size)')
        ax_8.set_ylabel('Log(Depth)')
        ax_8.grid(alpha=0.3)
    line_legend_1 = [mlines.Line2D([], [], color=colors_17['left'][1], linewidth=2, label='Left'), mlines.Line2D([], [], color=colors_17['center'][1], linewidth=2, label='Center'), mlines.Line2D([], [], color=colors_17['right'][1], linewidth=2, label='Right')]
    fig_34.legend(handles=line_legend_1, title='Robust Regression Line Color Legend', loc='upper center', bbox_to_anchor=(0.17, 0.85), ncol=3)
    plt.tight_layout()
    plt.show()
    return (norm,)


@app.cell
def _(df_14, mlines, np, plt):
    colors_18 = {'left': ('#436685', '#2A4765'), 'center': ('#bbcd78', '#8FA34F'), 'right': ('#8a2520', '#5E1815')}
    df_bsky_10 = df_14[df_14['platform'] == 'bsky'].copy()
    df_ts_10 = df_14[df_14['platform'] == 'ts'].copy()
    for df_platform_3 in [df_bsky_10, df_ts_10]:
        df_platform_3['log_size'] = np.log10(df_platform_3['size'])
        df_platform_3['log_breadth'] = np.log10(df_platform_3['breadth'])

    def kernel_regression_3(x_train, y_train, x_pred, bandwidth=0.5):
        """Performs Nadaraya-Watson Kernel Regression using Gaussian kernel."""
        y_pred = np.zeros_like(x_pred)
        for (i, x) in enumerate(x_pred):
            weights = np.exp(-(x_train - x) ** 2 / (2 * bandwidth ** 2))
            weights = weights / weights.sum()
            y_pred[i] = np.sum(weights * y_train)
        return y_pred
    (fig_35, axes_35) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    def plot_kernel_by_partisanship_1(df_platform, ax):
        for partisanship in ['left', 'center', 'right']:
            df_subset = df_platform[df_platform['partisanship'] == partisanship]
            if df_subset.empty:
                continue
            x_pred = np.linspace(df_subset['log_size'].min(), df_subset['log_size'].max(), 200)
            y_pred = kernel_regression_3(df_subset['log_size'].values, df_subset['log_breadth'].values, x_pred)
            (scatter_color, line_color) = colors_18[partisanship]
            ax.scatter(df_subset['log_size'], df_subset['log_breadth'], alpha=0.2, color=scatter_color)
            ax.plot(x_pred, y_pred, color=line_color, linewidth=2)
    plot_kernel_by_partisanship_1(df_bsky_10, axes_35[0])
    axes_35[0].set_title('BlueSky: Kernel Smoothed Breadth by Partisanship')
    plot_kernel_by_partisanship_1(df_ts_10, axes_35[1])
    axes_35[1].set_title('TruthSocial: Kernel Smoothed Breadth by Partisanship')
    for ax_9 in axes_35:
        ax_9.set_xlabel('Log(Size)')
        ax_9.set_ylabel('Log(Breadth)')
        ax_9.grid(alpha=0.3)
    line_legend_2 = [mlines.Line2D([], [], color=colors_18['left'][1], linewidth=2, label='Left'), mlines.Line2D([], [], color=colors_18['center'][1], linewidth=2, label='Center'), mlines.Line2D([], [], color=colors_18['right'][1], linewidth=2, label='Right')]
    fig_35.legend(handles=line_legend_2, title='Kernel Regression Line Color Legend', loc='upper center', bbox_to_anchor=(0.17, 0.85), ncol=3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_14, mlines, norm, np, plt, sm):
    colors_19 = {'left': ('#436685', '#2A4765'), 'center': ('#bbcd78', '#8FA34F'), 'right': ('#8a2520', '#5E1815')}
    df_bsky_11 = df_14[df_14['platform'] == 'bsky'].copy()
    df_ts_11 = df_14[df_14['platform'] == 'ts'].copy()
    for df_platform_4 in [df_bsky_11, df_ts_11]:
        df_platform_4['log_size'] = np.log10(df_platform_4['size'])
        df_platform_4['log_breadth'] = np.log10(df_platform_4['breadth'])

    def robust_fit_12(x, y):
        """Fits a robust regression model and returns sorted predictions and model."""
        x_const = sm.add_constant(x)
        model = sm.RLM(y, x_const, M=sm.robust.norms.HuberT()).fit()
        x_sorted = np.sort(x)
        x_pred = sm.add_constant(x_sorted)
        y_pred = model.predict(x_pred)
        return (x_sorted, y_pred, model)
    (fig_36, axes_36) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    def plot_robust_by_partisanship_1(df_platform, ax, platform_name):
        slopes = {}
        for partisanship in ['left', 'center', 'right']:
            df_subset = df_platform[df_platform['partisanship'] == partisanship]
            if df_subset.empty:
                continue
            (x_sorted, y_pred, model) = robust_fit_12(df_subset['log_size'].values, df_subset['log_breadth'].values)
            (slope, se) = (model.params[1], model.bse[1])
            slopes[partisanship] = (slope, se)
            (scatter_color, line_color) = colors_19[partisanship]
            ax.scatter(df_subset['log_size'], df_subset['log_breadth'], alpha=0.2, color=scatter_color)
            ax.plot(x_sorted, y_pred, color=line_color, linewidth=2, label=f'{partisanship.capitalize()} Fit')
        if 'left' in slopes and 'right' in slopes:
            (left_slope, left_se) = slopes['left']
            (right_slope, right_se) = slopes['right']
            z_score = (left_slope - right_slope) / np.sqrt(left_se ** 2 + right_se ** 2)
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            ax.text(min(df_platform['log_size']), max(df_platform['log_breadth']), f'Paternoster Z-test: Z={z_score:.2f}, p={p_value:.4f}\nLeft Slope: {left_slope:.4f}\nRight Slope: {right_slope:.4f}', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plot_robust_by_partisanship_1(df_bsky_11, axes_36[0], 'BlueSky')
    axes_36[0].set_title('BlueSky: Robust Regression Breadth by Partisanship')
    plot_robust_by_partisanship_1(df_ts_11, axes_36[1], 'TruthSocial')
    axes_36[1].set_title('TruthSocial: Robust Regression Breadth by Partisanship')
    for ax_10 in axes_36:
        ax_10.set_xlabel('Log(Size)')
        ax_10.set_ylabel('Log(Breadth)')
        ax_10.grid(alpha=0.3)
    line_legend_3 = [mlines.Line2D([], [], color=colors_19['left'][1], linewidth=2, label='Left'), mlines.Line2D([], [], color=colors_19['center'][1], linewidth=2, label='Center'), mlines.Line2D([], [], color=colors_19['right'][1], linewidth=2, label='Right')]
    fig_36.legend(handles=line_legend_3, title='Robust Regression Line Color Legend', loc='upper center', bbox_to_anchor=(0.17, 0.85), ncol=3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Repost""")
    return


@app.cell
def _(np, pd, plt, rp_bsky_df_4, rp_ts_df_3, sns):
    partisan_colors_4 = {'left': '#436685', 'center': '#bbcd78', 'right': '#8a2520'}

    def expected_depth_1(rp_df):
        rp_df['log_size'] = np.log10(rp_df['size'].replace(0, np.nan))
        rp_df['log_depth'] = np.log10(rp_df['max_depth'].replace(0, np.nan))
        rp_df['size_bin'] = pd.qcut(rp_df['log_size'], q=10, duplicates='drop')
        depth_median = rp_df.groupby('size_bin')['log_depth'].median()
        rp_df['expected_depth'] = rp_df['size_bin'].map(depth_median).astype(float)
        return rp_df
    bsky_rp_df = expected_depth_1(rp_bsky_df_4)
    ts_rp_df = expected_depth_1(rp_ts_df_3)

    def identify_outliers_1(rp_df):
        return rp_df[(rp_df['size'] > 100) & (rp_df['log_depth'] < rp_df['expected_depth'] - 0.5)]
    bsky_outliers_1 = identify_outliers_1(bsky_rp_df)
    ts_outliers_2 = identify_outliers_1(ts_rp_df)
    (fig_37, axes_37) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    sns.scatterplot(data=bsky_rp_df, x='size', y='max_depth', ax=axes_37[0], color='gray', alpha=0.3)
    sns.scatterplot(data=ts_rp_df, x='size', y='max_depth', ax=axes_37[1], color='gray', alpha=0.3)

    def plot_outliers_with_colors_2(data, ax):
        for (partisanship, color) in partisan_colors_4.items():
            subset = data[data['partisanship'] == partisanship]
            sns.scatterplot(data=subset, x='size', y='max_depth', ax=ax, color=color, label=partisanship.capitalize(), s=50, edgecolor='black', alpha=0.8)
    plot_outliers_with_colors_2(bsky_outliers_1, axes_37[0])
    plot_outliers_with_colors_2(ts_outliers_2, axes_37[1])
    for ax_11 in axes_37:
        ax_11.set(xscale='log', yscale='log')
        ax_11.legend(title='Partisanship (Outliers)', loc='upper left')
        ax_11.set_xlabel('Size')
        ax_11.set_ylabel('Depth')
    axes_37[0].set_title('BlueSky: Size vs Depth (Outliers Highlighted)')
    axes_37[1].set_title('TruthSocial: Size vs Depth (Outliers Highlighted)')
    plt.tight_layout()
    plt.show()
    return bsky_rp_df, ts_rp_df


@app.cell
def _(bsky_rp_df, plt, rp_bsky_df_4, rp_ts_df_3, sns, ts_rp_df):
    partisan_colors_5 = {'left': '#436685', 'center': '#bbcd78', 'right': '#8a2520'}
    rp_ts_df_3['breadth_size_ratio'] = rp_ts_df_3['breadth'] / ts_rp_df['size']
    rp_bsky_df_4['breadth_size_ratio'] = rp_bsky_df_4['breadth'] / bsky_rp_df['size']
    rp_bsky_df_4['outlier'] = 'no'
    rp_bsky_df_4.loc[(bsky_rp_df['size'] > 100) & (rp_bsky_df_4['breadth_size_ratio'] < 0.1), 'outlier'] = 'yes'
    rp_ts_df_3['outlier'] = 'no'
    rp_ts_df_3.loc[(rp_ts_df_3['size'] > 1000) & (rp_ts_df_3['breadth_size_ratio'] < 0.1), 'outlier'] = 'yes'
    (fig_38, axes_38) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    bsky_rp_df_1 = rp_bsky_df_4
    ts_rp_df_1 = rp_ts_df_3
    sns.scatterplot(data=bsky_rp_df_1[bsky_rp_df_1['outlier'] == 'no'], x='size', y='breadth', ax=axes_38[0], color='gray', alpha=0.3)
    sns.scatterplot(data=ts_rp_df_1[ts_rp_df_1['outlier'] == 'no'], x='size', y='breadth', ax=axes_38[1], color='gray', alpha=0.3)

    def plot_outliers_with_colors_3(data, ax):
        for (partisanship, color) in partisan_colors_5.items():
            subset = data[data['partisanship'] == partisanship]
            sns.scatterplot(data=subset, x='size', y='breadth', ax=ax, color=color, label=partisanship.capitalize(), s=50, edgecolor='black', alpha=0.8)
    plot_outliers_with_colors_3(bsky_rp_df_1[bsky_rp_df_1['outlier'] == 'yes'], axes_38[0])
    plot_outliers_with_colors_3(ts_rp_df_1[ts_rp_df_1['outlier'] == 'yes'], axes_38[1])
    for ax_12 in axes_38:
        ax_12.set(xscale='log', yscale='log')
        ax_12.set_xlabel('Size')
        ax_12.set_ylabel('Breadth')
    (handles_12, labels_14) = axes_38[1].get_legend_handles_labels()
    axes_38[0].get_legend().remove()
    axes_38[1].get_legend().remove()
    fig_38.legend(handles_12, labels_14, title='Partisanship (Outliers)', loc='upper center', ncol=len(labels_14), bbox_to_anchor=(0.5, 1.05))
    axes_38[0].set_title('BlueSky: Size vs Breadth (Outliers Highlighted)')
    axes_38[1].set_title('TruthSocial: Size vs Breadth (Outliers Highlighted)')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Size vs. Depth and Breadth""")
    return


@app.cell
def _(f_1, np, pd, plt, rp_bsky_df_4, rp_ts_df_3, sm):
    colors_20 = {'bsky_scatter': '#007FFF', 'bsky_rlm': '#0056A3', 'ts_scatter': '#FFD700', 'ts_rlm': '#C49A00'}
    df_15 = pd.concat([rp_bsky_df_4, rp_ts_df_3], ignore_index=True)
    df_bsky_12 = df_15[df_15['platform'] == 'bsky'].copy()
    df_ts_12 = df_15[df_15['platform'] == 'ts'].copy()
    df_bsky_12['log_size'] = np.log10(df_bsky_12['size'])
    df_bsky_12['log_breadth'] = np.log10(df_bsky_12['breadth'])
    df_ts_12['log_size'] = np.log10(df_ts_12['size'])
    df_ts_12['log_breadth'] = np.log10(df_ts_12['breadth'])

    def robust_fit_13(x, y):
        """Fits a robust regression model and returns sorted predictions."""
        x_const = sm.add_constant(x)
        model = sm.RLM(y, x_const, M=sm.robust.norms.HuberT())
        results = model.fit()
        x_sorted = np.sort(x)
        x_pred = sm.add_constant(x_sorted)
        y_pred = results.predict(x_pred)
        return (x_sorted, y_pred, results)
    (bsky_x_5, bsky_y_5, model_bsky_5) = robust_fit_13(df_bsky_12['log_size'].values, df_bsky_12['log_breadth'].values)
    (ts_x_5, ts_y_5, model_ts_5) = robust_fit_13(df_ts_12['log_size'].values, df_ts_12['log_breadth'].values)
    bsky_slope_5 = model_bsky_5.params[1]
    ts_slope_5 = model_ts_5.params[1]

    def chow_test_5(x1, y1, x2, y2):
        """Performs the Chow test to compare regression slopes between two datasets."""
        (x1_const, x2_const) = (sm.add_constant(x1), sm.add_constant(x2))
        model1 = sm.OLS(y1, x1_const).fit()
        model2 = sm.OLS(y2, x2_const).fit()
        x_combined = np.concatenate([x1, x2])
        y_combined = np.concatenate([y1, y2])
        x_combined_const = sm.add_constant(x_combined)
        model_combined = sm.OLS(y_combined, x_combined_const).fit()
        SSR_combined = np.sum(model_combined.resid ** 2)
        SSR1 = np.sum(model1.resid ** 2)
        SSR2 = np.sum(model2.resid ** 2)
        (n1, n2) = (len(y1), len(y2))
        k = 2
        numerator = (SSR_combined - (SSR1 + SSR2)) / k
        denominator = (SSR1 + SSR2) / (n1 + n2 - 2 * k)
        if denominator == 0:
            (chow_stat, p_value) = (np.nan, 1.0)
        else:
            chow_stat = numerator / denominator
            p_value = 1 - f_1.cdf(chow_stat, k, n1 + n2 - 2 * k)
        return (chow_stat, p_value)
    (chow_stat_5, p_value_19) = chow_test_5(df_bsky_12['log_size'].values, df_bsky_12['log_breadth'].values, df_ts_12['log_size'].values, df_ts_12['log_breadth'].values)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(df_bsky_12['log_size'], df_bsky_12['log_breadth'], alpha=0.1, color=colors_20['bsky_scatter'], label='BlueSky Data')
    plt.scatter(df_ts_12['log_size'], df_ts_12['log_breadth'], alpha=0.1, color=colors_20['ts_scatter'], label='TruthSocial Data')
    plt.plot(bsky_x_5, bsky_y_5, color=colors_20['bsky_rlm'], linewidth=2, label='BlueSky Robust Fit')
    plt.plot(ts_x_5, ts_y_5, color=colors_20['ts_rlm'], linewidth=2, label='TruthSocial Robust Fit')
    plt.xlabel('Size')
    plt.ylabel('Breadth')
    plt.title('Robust Regression of Breadth Across Platformin Repost Network')
    plt.text(min(df_bsky_12['log_size']), max(df_bsky_12['log_breadth']), f'BlueSky Slope: {bsky_slope_5:.4f}\nTruthSocial Slope: {ts_slope_5:.4f}', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(np, pd, plt, rp_bsky_df_4, rp_ts_df_3):
    colors_21 = {'bsky_scatter': '#007FFF', 'bsky_kernel': '#0056A3', 'ts_scatter': '#FFD700', 'ts_kernel': '#C49A00'}
    df_16 = pd.concat([rp_bsky_df_4, rp_ts_df_3], ignore_index=True)
    df_bsky_13 = df_16[df_16['platform'] == 'bsky'].copy()
    df_ts_13 = df_16[df_16['platform'] == 'ts'].copy()
    df_bsky_13['log_size'] = np.log10(df_bsky_13['size'])
    df_bsky_13['log_breadth'] = np.log10(df_bsky_13['breadth'])
    df_ts_13['log_size'] = np.log10(df_ts_13['size'])
    df_ts_13['log_breadth'] = np.log10(df_ts_13['breadth'])

    def kernel_regression_4(x_train, y_train, x_pred, bandwidth=0.5):
        """Performs Nadaraya-Watson Kernel Regression using Gaussian kernel."""
        y_pred = np.zeros_like(x_pred)
        for (i, x) in enumerate(x_pred):
            weights = np.exp(-(x_train - x) ** 2 / (2 * bandwidth ** 2))
            weights = weights / weights.sum()
            y_pred[i] = np.sum(weights * y_train)
        return y_pred

    def bootstrap_ci_kernel_1(x_train, y_train, x_pred, bandwidth=0.5, n_bootstrap=1000, alpha=0.05):
        """Computes bootstrap confidence intervals for kernel regression."""
        bootstrap_preds = np.zeros((n_bootstrap, len(x_pred)))
        n = len(y_train)
        for i in range(n_bootstrap):
            sample_idx = np.random.choice(n, n, replace=True)
            (x_sample, y_sample) = (x_train[sample_idx], y_train[sample_idx])
            bootstrap_preds[i, :] = kernel_regression_4(x_sample, y_sample, x_pred, bandwidth)
        lower_ci = np.percentile(bootstrap_preds, 100 * alpha / 2, axis=0)
        upper_ci = np.percentile(bootstrap_preds, 100 * (1 - alpha / 2), axis=0)
        return (lower_ci, upper_ci)
    x_pred_bsky_2 = np.linspace(df_bsky_13['log_size'].min(), df_bsky_13['log_size'].max(), 200)
    x_pred_ts_2 = np.linspace(df_ts_13['log_size'].min(), df_ts_13['log_size'].max(), 200)
    bsky_y_pred_2 = kernel_regression_4(df_bsky_13['log_size'].values, df_bsky_13['log_breadth'].values, x_pred_bsky_2)
    ts_y_pred_2 = kernel_regression_4(df_ts_13['log_size'].values, df_ts_13['log_breadth'].values, x_pred_ts_2)
    (bsky_lower_ci_1, bsky_upper_ci_1) = bootstrap_ci_kernel_1(df_bsky_13['log_size'].values, df_bsky_13['log_breadth'].values, x_pred_bsky_2)
    (ts_lower_ci_1, ts_upper_ci_1) = bootstrap_ci_kernel_1(df_ts_13['log_size'].values, df_ts_13['log_breadth'].values, x_pred_ts_2)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(df_bsky_13['log_size'], df_bsky_13['log_breadth'], alpha=0.1, color=colors_21['bsky_scatter'], label='BlueSky Data')
    plt.scatter(df_ts_13['log_size'], df_ts_13['log_breadth'], alpha=0.1, color=colors_21['ts_scatter'], label='TruthSocial Data')
    plt.plot(x_pred_bsky_2, bsky_y_pred_2, color=colors_21['bsky_kernel'], linewidth=2, label='BlueSky Kernel Fit')
    plt.plot(x_pred_ts_2, ts_y_pred_2, color=colors_21['ts_kernel'], linewidth=2, label='TruthSocial Kernel Fit')
    plt.fill_between(x_pred_bsky_2, bsky_lower_ci_1, bsky_upper_ci_1, color=colors_21['bsky_kernel'], alpha=0.2, label='BlueSky 95% CI')
    plt.fill_between(x_pred_ts_2, ts_lower_ci_1, ts_upper_ci_1, color=colors_21['ts_kernel'], alpha=0.2, label='TruthSocial 95% CI')
    plt.xlabel('Size')
    plt.ylabel('Breadth')
    plt.title('Kernel Regression with Confidence Intervals')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(np, pd, plt, rp_bsky_df_4, rp_ts_df_3, sm):
    colors_22 = {'bsky_scatter': '#007FFF', 'bsky_rlm': '#0056A3', 'ts_scatter': '#FFD700', 'ts_rlm': '#C49A00'}
    df_17 = pd.concat([rp_bsky_df_4, rp_ts_df_3], ignore_index=True)
    df_bsky_14 = df_17[df_17['platform'] == 'bsky'].copy()
    df_ts_14 = df_17[df_17['platform'] == 'ts'].copy()
    df_bsky_14['log_size'] = np.log10(df_bsky_14['size'])
    df_bsky_14['log_max_depth'] = np.log10(df_bsky_14['max_depth'] + 1)
    df_ts_14['log_size'] = np.log10(df_ts_14['size'])
    df_ts_14['log_max_depth'] = np.log10(df_ts_14['max_depth'] + 1)

    def robust_fit_with_ci(x, y, n_bootstrap=1000, alpha=0.05):
        """Fits robust regression and computes bootstrap confidence intervals."""
        x_const = sm.add_constant(x)
        model = sm.RLM(y, x_const, M=sm.robust.norms.HuberT())
        results = model.fit()
        x_sorted = np.sort(x)
        x_pred = sm.add_constant(x_sorted)
        y_pred = results.predict(x_pred)
        preds = np.zeros((n_bootstrap, len(x_sorted)))
        n = len(y)
        for i in range(n_bootstrap):
            sample_idx = np.random.choice(n, n, replace=True)
            (x_sample, y_sample) = (x[sample_idx], y[sample_idx])
            x_sample_const = sm.add_constant(x_sample)
            try:
                resample_model = sm.RLM(y_sample, x_sample_const, M=sm.robust.norms.HuberT()).fit()
                preds[i, :] = resample_model.predict(x_pred)
            except:
                preds[i, :] = np.nan
        lower = np.nanpercentile(preds, 100 * alpha / 2, axis=0)
        upper = np.nanpercentile(preds, 100 * (1 - alpha / 2), axis=0)
        return (x_sorted, y_pred, lower, upper)
    (bsky_x_6, bsky_y_6, bsky_lower, bsky_upper) = robust_fit_with_ci(df_bsky_14['log_size'].values, df_bsky_14['log_max_depth'].values)
    (ts_x_6, ts_y_6, ts_lower, ts_upper) = robust_fit_with_ci(df_ts_14['log_size'].values, df_ts_14['log_max_depth'].values)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(df_bsky_14['log_size'], df_bsky_14['log_max_depth'], alpha=0.1, color=colors_22['bsky_scatter'], label='BlueSky Data')
    plt.scatter(df_ts_14['log_size'], df_ts_14['log_max_depth'], alpha=0.1, color=colors_22['ts_scatter'], label='TruthSocial Data')
    plt.plot(bsky_x_6, bsky_y_6, color=colors_22['bsky_rlm'], linewidth=2, label='BlueSky Robust Fit')
    plt.plot(ts_x_6, ts_y_6, color=colors_22['ts_rlm'], linewidth=2, label='TruthSocial Robust Fit')
    plt.fill_between(bsky_x_6, bsky_lower, bsky_upper, color=colors_22['bsky_rlm'], alpha=0.2, label='BlueSky 95% CI')
    plt.fill_between(ts_x_6, ts_lower, ts_upper, color=colors_22['ts_rlm'], alpha=0.2, label='TruthSocial 95% CI')
    plt.xlabel('Size')
    plt.ylabel('Depth')
    plt.title('Robust Regression with Confidence Intervals of Depth in Repost Network')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(np, pd, plt, rp_bsky_df_4, rp_ts_df_3):
    colors_23 = {'bsky_scatter': '#007FFF', 'bsky_kernel': '#0056A3', 'ts_scatter': '#FFD700', 'ts_kernel': '#C49A00'}
    df_18 = pd.concat([rp_bsky_df_4, rp_ts_df_3], ignore_index=True)
    df_bsky_15 = df_18[df_18['platform'] == 'bsky'].copy()
    df_ts_15 = df_18[df_18['platform'] == 'ts'].copy()
    df_bsky_15['log_size'] = np.log10(df_bsky_15['size'])
    df_bsky_15['log_max_depth'] = np.log10(df_bsky_15['max_depth'] + 1)
    df_ts_15['log_size'] = np.log10(df_ts_15['size'])
    df_ts_15['log_max_depth'] = np.log10(df_ts_15['max_depth'] + 1)

    def kernel_regression_5(x_train, y_train, x_pred, bandwidth=0.5):
        """Performs Nadaraya-Watson Kernel Regression using Gaussian kernel."""
        y_pred = np.zeros_like(x_pred)
        for (i, x) in enumerate(x_pred):
            weights = np.exp(-(x_train - x) ** 2 / (2 * bandwidth ** 2))
            weights = weights / weights.sum()
            y_pred[i] = np.sum(weights * y_train)
        return y_pred
    x_pred_bsky_3 = np.linspace(df_bsky_15['log_size'].min(), df_bsky_15['log_size'].max(), 200)
    x_pred_ts_3 = np.linspace(df_ts_15['log_size'].min(), df_ts_15['log_size'].max(), 200)
    bsky_y_pred_3 = kernel_regression_5(df_bsky_15['log_size'].values, df_bsky_15['log_max_depth'].values, x_pred_bsky_3)
    ts_y_pred_3 = kernel_regression_5(df_ts_15['log_size'].values, df_ts_15['log_max_depth'].values, x_pred_ts_3)
    plt.figure(figsize=(8, 5), dpi=150)
    plt.scatter(df_bsky_15['log_size'], df_bsky_15['log_max_depth'], alpha=0.1, color=colors_23['bsky_scatter'], label='BlueSky Data')
    plt.scatter(df_ts_15['log_size'], df_ts_15['log_max_depth'], alpha=0.1, color=colors_23['ts_scatter'], label='TruthSocial Data')
    plt.plot(x_pred_bsky_3, bsky_y_pred_3, color=colors_23['bsky_kernel'], linewidth=2, label='BlueSky Kernel Fit')
    plt.plot(x_pred_ts_3, ts_y_pred_3, color=colors_23['ts_kernel'], linewidth=2, label='TruthSocial Kernel Fit')
    plt.xlabel('Size')
    plt.ylabel('max_depth')
    plt.title('Kernel Regression of max_depth Across Platforms')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(ts_df_5, ts_raw):
    for ids in ts_df_5.loc[(ts_df_5['size'] > 1000) & (ts_df_5['breadth_size_ratio'] > 0.5), 'index']:
        for post in ts_raw:
            if post['_id'] == str(ids):
                print(ids, post['account']['username'])
    return


@app.cell
def _(pprint, ts_raw):
    for post_1 in ts_raw:
        if post_1['_id'] == '112621173765349628':
            pprint(post_1)
            break
    return


@app.cell
def _(ts_raw):
    for post_2 in ts_raw:
        if post_2['_id'] == '112621173765349628':
            print(post_2['account'])
            break
    return


@app.cell
def _(ts_df_5):
    ts_df_5['depth_size_ratio'] = ts_df_5['max_depth'] / ts_df_5['size']
    return


app._unparsable_cell(
    r"""
    ts_df.loc[(ts_df['size']>500) & (ts_df['depth_size_ratio']>0.1), :].sort_values('depth_size_ratio', ascending=).head(10)
    """,
    name="_"
)


@app.cell
def _(plt, rp_bsky_df_4, rp_ts_df_3):
    (fig_39, axes_39) = plt.subplots(1, 2, figsize=(14, 6))
    axes_39[0].scatter(rp_bsky_df_4['log_size'], rp_bsky_df_4['max_depth'], label='bsky', alpha=0.1)
    axes_39[1].scatter(rp_ts_df_3['log_size'], rp_ts_df_3['max_depth'], label='ts', alpha=0.1)
    axes_39[0].set_xlabel('log_size')
    axes_39[0].set_ylabel('max_depth')
    axes_39[1].set_xlabel('log_size')
    axes_39[1].set_ylabel('max_depth')
    axes_39[0].set_title('bsky')
    axes_39[1].set_title('ts')
    plt.show()
    return


@app.cell
def _(plt, rp_bsky_df_4, rp_ts_df_3):
    (fig_40, axes_40) = plt.subplots(1, 2, figsize=(14, 6))
    axes_40[0].scatter(rp_bsky_df_4['log_size'], rp_bsky_df_4['breadth'], label='bsky', alpha=0.1)
    axes_40[1].scatter(rp_ts_df_3['log_size'], rp_ts_df_3['breadth'], label='ts', alpha=0.1)
    axes_40[0].set_xlabel('log_size')
    axes_40[0].set_ylabel('breadth')
    axes_40[1].set_xlabel('log_size')
    axes_40[1].set_ylabel('breadth')
    axes_40[0].set_title('bsky')
    axes_40[1].set_title('ts')
    plt.show()
    return


@app.cell
def _(rp_ts_df_3):
    rp_ts_df_3['breadth_size_ratio'] = rp_ts_df_3['breadth'] / rp_ts_df_3['size']
    rp_ts_df_3.sort_values(['size', 'breadth_size_ratio'], ascending=False).head(10)
    return


@app.cell
def _(ts_raw):
    from pprint import pprint
    for post_3 in ts_raw:
        if post_3['_id'] == '112532453963389112':
            pprint(post_3)
            break
    return (pprint,)


@app.cell
def _(rp_ts_df_3, ts_raw):
    for ids_1 in rp_ts_df_3.sort_values(['size', 'breadth_size_ratio'], ascending=False).head(20)['index']:
        for post_4 in ts_raw:
            if post_4['_id'] == str(ids_1):
                print(ids_1, post_4['account']['username'])
                break
    return (ids_1,)


@app.cell
def _(ids_1):
    ids_1
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
