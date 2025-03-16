import marimo

__generated_with = "0.11.20"
app = marimo.App(width="full")


@app.cell
def _(np, tqdm):
    import cudf
    import cupy as cp
    import networkx as nx
    import cugraph as cg
    edgeType2id = {'repost': 0, 'directed': 1, 'indirected': 2}
    id2edgeType = {v: k for (k, v) in edgeType2id.items()}

    def convert_node_ids_to_int(graph):
        """
        Convert node IDs in a graph to integers for fast processing.

        - Assigns a unique integer ID to each node.
        - Returns a mapping dictionary for original IDs.
        - Replaces edges in the graph with integer-based edges.

        :param graph: NetworkX graph with any node identifiers.
        :return: New graph with integer node IDs, mapping dictionary
        """
        node_mapping = {node: i for (i, node) in enumerate(graph.nodes())}
        new_graph = nx.DiGraph()
        for (u, v, data) in graph.edges(data=True):
            new_graph.add_edge(node_mapping[u], node_mapping[v], **data)
        return (new_graph, node_mapping)

    def build_edge_lookup_dict(graph):
        """
        Build a fast edge lookup dictionary using integer node IDs and integer edge type IDs.

        :param graph: The directed graph with integer-based node IDs.
        :param node_mapping: Dictionary mapping original node IDs to integers.
        :param edge_type_mapping: Dictionary mapping edge types to integers.
        :return: Dictionary lookup { (src, dst): edge_type }
        """
        edge_lookup = {}
        for (u, v, data) in graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_lookup[u, v] = edge_type
        return edge_lookup

    def get_edge_type(a, b, edge_lookup_dict):
        """
        Fetch integer edge type from dictionary lookup.

        :param a: Integer node ID (source)
        :param b: Integer node ID (destination)
        :param edge_lookup_dict: Dictionary {(src, dst): edge_type}
        :return: Integer edge type ID or -1 if not found.
        """
        return edge_lookup_dict.get((a, b), -1)

    def extract_3node_subtrees(tree):
        """Extract all 3-node tree motifs efficiently."""
        subtrees = []
        for node in tree.nodes():
            children = list(tree.successors(node))
            if len(children) >= 2:
                subtrees.extend([(node, children[i], children[j]) for i in range(len(children)) for j in range(i + 1, len(children))])
            for child in children:
                grandchildren = list(tree.successors(child))
                subtrees.extend([(node, child, grandchild) for grandchild in grandchildren])
        return subtrees
    motif_mapping = {('star', 'repost', 'repost'): 0, ('star', 'repost', 'directed'): 1, ('star', 'repost', 'indirected'): 2, ('star', 'directed', 'directed'): 3, ('star', 'directed', 'indirected'): 4, ('star', 'indirected', 'indirected'): 5, ('star', 'indirected', 'directed'): 6, ('chain', 'repost', 'repost'): 7, ('chain', 'directed', 'repost'): 8, ('chain', 'indirected', 'repost'): 9, ('chain', 'directed', 'directed'): 10, ('chain', 'directed', 'indirected'): 11, ('chain', 'indirected', 'indirected'): 12, ('chain', 'indirected', 'directed'): 13}

    def classify_motif(subtree, edge_lookup_dict):
        """
        Classify 3-node subtree motifs using dictionary-based edge types.

        :param subtree: (a, b, c) tuple of integer node IDs.
        :param edge_lookup_dict: Dictionary {(src, dst): edge_type}
        :return: Integer motif ID.
        """
        (a, b, c) = subtree
        edge1 = get_edge_type(a, b, edge_lookup_dict)
        edge2 = get_edge_type(b, c, edge_lookup_dict) if (b, c) in edge_lookup_dict else get_edge_type(a, c, edge_lookup_dict)
        structure = 'star' if (a, c) in edge_lookup_dict else 'chain'
        return motif_mapping.get((structure, edge1, edge2), -1)
    subtree_kernel = cp.RawKernel('\nextern "C" __global__\nvoid classify_subtrees(int *motifs, int N, int *counts) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx >= N) return;\n    atomicAdd(&counts[motifs[idx]], 1);\n}\n', 'classify_subtrees')

    def prepare_edge_array(graph, edgeType2id):
        """
        Convert graph edges into a structured NumPy array with edge types.

        :param graph: NetworkX directed graph
        :return: NumPy structured array containing (source, target, edge_type)
        """
        edge_list = [(u, v, edgeType2id.get(graph[u][v]['type'])) for (u, v) in graph.edges()]
        dtype = [('src', np.int32), ('dst', np.int32), ('type', np.int32)]
        return np.array(edge_list, dtype=dtype)

    def vectorized_edge_swap_with_progress(edges, num_swaps, batch_size=10000):
        """
        Perform edge swaps efficiently using vectorized NumPy operations with tqdm progress tracking.

        :param edges: NumPy structured array of edges [(src, dst, type)]
        :param num_swaps: Total number of swaps to perform
        :param batch_size: Number of swaps per batch (for tqdm updates)
        :return: Modified edge array with swapped edges
        """
        num_edges = len(edges)
        swaps_done = 0
        with tqdm(total=num_swaps, desc='Swapping Edges', unit='swap') as pbar:
            while swaps_done < num_swaps:
                swaps_to_do = min(batch_size, num_swaps - swaps_done)
                idx1 = np.random.choice(num_edges, swaps_to_do, replace=True)
                idx2 = np.random.choice(num_edges, swaps_to_do, replace=True)
                mask = idx1 != idx2
                (idx1, idx2) = (idx1[mask], idx2[mask])
                (src1, dst1, type1) = (edges['src'][idx1], edges['dst'][idx1], edges['type'][idx1])
                (src2, dst2, type2) = (edges['src'][idx2], edges['dst'][idx2], edges['type'][idx2])
                valid_mask = (src1 != src2) & (dst1 != dst2) & (src1 != dst2) & (src2 != dst1)
                (edges['dst'][idx1[valid_mask]], edges['dst'][idx2[valid_mask]]) = (dst2[valid_mask], dst1[valid_mask])
                (edges['type'][idx1[valid_mask]], edges['type'][idx2[valid_mask]]) = (type2[valid_mask], type1[valid_mask])
                swaps_done = swaps_done + len(idx1[valid_mask])
                pbar.update(len(idx1[valid_mask]))
        return edges

    def directed_edge_swap_fully_vectorized(graph, edgeType2id, num_swaps=500000, batch_size=10000):
        """
        Fully vectorized edge swapping function using NumPy for high-speed execution with tqdm progress bar.

        :param graph: Input directed graph (NetworkX DiGraph)
        :param num_swaps: Number of swaps to attempt
        :param batch_size: Number of swaps per batch for tqdm updates
        :return: A new randomized graph with preserved edge types
        """
        G = graph.copy()
        edge_array = prepare_edge_array(G, edgeType2id)
        edge_array = vectorized_edge_swap_with_progress(edge_array, num_swaps, batch_size=batch_size)
        G.clear_edges()
        for (src, dst, edge_type) in edge_array:
            edge_type = id2edgeType[edge_type]
            G.add_edge(src, dst, type=edge_type)
        return G
    return (
        build_edge_lookup_dict,
        cg,
        classify_motif,
        convert_node_ids_to_int,
        cp,
        cudf,
        directed_edge_swap_fully_vectorized,
        edgeType2id,
        extract_3node_subtrees,
        get_edge_type,
        id2edgeType,
        motif_mapping,
        nx,
        prepare_edge_array,
        subtree_kernel,
        vectorized_edge_swap_with_progress,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Start building the network for the overall analysis""")
    return


@app.cell
def _():
    # import repost and following data
    import json
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from cascade_analysis import InformationCascadeGraph
    return InformationCascadeGraph, datetime, json, np, pd


@app.cell
def _(json):
    with open("data/bsky_reposts_new.json") as f:
        bsky_repost = json.load(f)

    with open("data/bsky_follows.json") as f:
        bsky_follow = json.load(f)

    with open('data/bsky_post_to_label.json', 'r') as f:
        bsky_post_to_label = json.load(f)
    return bsky_follow, bsky_post_to_label, bsky_repost, f


@app.cell
def _():
    from importlib import reload
    import cascade_analysis

    reload(cascade_analysis)
    return cascade_analysis, reload


@app.cell
def _(bsky_follow, bsky_post_to_label, bsky_repost):
    from collections import defaultdict
    from itertools import chain

    original_list = bsky_follow

    # Use a defaultdict to store sets of DIDs.
    merged = defaultdict(set)

    # chain.from_iterable(...) flattens out the "dict.items()" across the list
    for key, records in chain.from_iterable(item.items() for item in original_list):
        # 'records' is the list of dicts. We update the set with the "did" values.
        merged[key].update(r["did"] for r in records)

    # Convert to a regular dict if desired:
    merged_dict = dict(merged)

    ideology_map = {}
    for post in bsky_repost:
        post_id = post['_id']
        text = post['record']['text']
        ideology = bsky_post_to_label.get(text, "center")
        if ideology == 'lean left':
            ideology = 'left'
        elif ideology == 'lean right':
            ideology = 'right'
        ideology_map[post_id] = ideology
    return (
        chain,
        defaultdict,
        ideology,
        ideology_map,
        key,
        merged,
        merged_dict,
        original_list,
        post,
        post_id,
        records,
        text,
    )


@app.cell
def _(bsky_repost, cascade_analysis, ideology_map, merged_dict):
    cascade_graph = cascade_analysis.InformationCascadeGraph(
        bsky_repost, merged_dict, ideology_map=ideology_map, platform="bsky"
    )
    return (cascade_graph,)


@app.cell
def _(cascade_graph):
    reposts_graph = cascade_graph.build_repost_graph()
    return (reposts_graph,)


@app.cell
def _(reposts_graph):
    reposts_graph.number_of_edges()
    return


@app.cell
def _(reposts_graph):
    reposts_graph.number_of_nodes()
    return


@app.cell
def _(cascade_graph):
    reply_graph = cascade_graph.build_reply_graph()
    return (reply_graph,)


@app.cell
def _(reply_graph):
    reply_graph.number_of_edges()
    return


@app.cell
def _(reply_graph):
    reply_graph.number_of_nodes()
    return


@app.cell
def _(nx, reply_graph, reposts_graph):
    # combine two graphs using the most naive way 
    combined_graph = nx.compose(reposts_graph, reply_graph)
    return (combined_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Null Models""")
    return


@app.cell
def _(np):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm 
    z_scores = np.array([148.51, -86.2871, -83.989, -36.0387, -69.543, -39.359, 161.779, -186.289, 158.127, 74.8111, -151.381, 3.92459, 161.456])

    # Example data (Replace with your actual z_scores)
    motif_ids = np.arange(13)


    # Step 1: Calculate two-tailed p-values from Z-scores
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))  # Two-tailed p-value calculation

    # Step 2: Define colors for points based on significance
    colors = ["#90a0c8" if p < 0.05 else "#f4c28f" for p in p_values]

    # Step 3: Set Seaborn style and color palette
    sns.set_theme(style="whitegrid")  # Beautiful modern theme

    # Step 4: Create the figure
    plt.figure(figsize=(10, 6), dpi=300)

    # Plot smooth line connecting points
    sns.lineplot(x=motif_ids, y=z_scores, color="gray", linewidth=2, linestyle="--", alpha=0.7)

    # Scatter plot for Z-scores with significance-based coloring
    sns.scatterplot(x=motif_ids, y=z_scores, hue=z_scores < 0, palette={True: "#90a0c8", False: "#f4c28f"}, s=120, edgecolor="black", legend=False)

    # Step 5: Add significance threshold line
    plt.axhline(y=2, color='black', linestyle='dashed', linewidth=1.5, label="Z = 2 (Significance Threshold)")

    # Step 6: Labels and title
    plt.xlabel("Motif ID", fontsize=13)
    plt.ylabel("Z-score", fontsize=13)
    plt.title("Significant Overrepresented Motifs", fontsize=15, fontweight="bold")

    # Step 7: Customize axes, ticks, and grid
    plt.xticks(motif_ids, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)

    # Step 8: Show legend and final plot
    plt.legend()
    plt.show()
    return colors, motif_ids, norm, p_values, plt, sns, z_scores


@app.cell
def _(norm, np, plt, sns):
    z_scores_1 = np.array([3.20728, 10.1647, -21.2754, 15.8421, -12.1197, -1.16916, 39.1044, -10.9542, 0.214503, 13.5716, -44.846, -7.57559, 47.2986])
    motif_ids_1 = np.arange(13)
    p_values_1 = 2 * (1 - norm.cdf(np.abs(z_scores_1)))
    colors_1 = ['#90a0c8' if p < 0.05 else '#f4c28f' for p in p_values_1]
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 6), dpi=300)
    sns.lineplot(x=motif_ids_1, y=z_scores_1, color='gray', linewidth=2, linestyle='--', alpha=0.7)
    sns.scatterplot(x=motif_ids_1, y=z_scores_1, hue=z_scores_1 < 0, palette={True: '#90a0c8', False: '#f4c28f'}, s=120, edgecolor='black', legend=False)
    plt.axhline(y=2, color='black', linestyle='dashed', linewidth=1.5, label='Z = 2 (Significance Threshold)')
    plt.xlabel('Motif ID', fontsize=13)
    plt.ylabel('Z-score', fontsize=13)
    plt.title('Significant Overrepresented Motifs', fontsize=15, fontweight='bold')
    plt.xticks(motif_ids_1, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.show()
    return colors_1, motif_ids_1, p_values_1, z_scores_1


@app.cell
def _(norm, np, plt):
    z_scores2 = np.array([3.20728, 10.1647, -21.2754, 15.8421, -12.1197, -1.16916, 39.1044, -10.9542, 0.214503, 13.5716, -44.846, -7.57559, 47.2986])
    motif_ids_2 = np.arange(13)
    z_scores1 = np.array([41.4376, -29.1637, -26.3728, -12.347, -33.526, -11.219, 57.8398, -47.8141, 40.985, 9.88934, -38.2146, -5.17001, 48.4784])
    grey_color = '#d3d3d3'
    p_values1 = 2 * (1 - norm.cdf(np.abs(z_scores1)))
    p_values2 = 2 * (1 - norm.cdf(np.abs(z_scores2)))
    neg_color = '#90a0c8'
    pos_color = '#f4c28f'
    colors1 = [pos_color if p < 0.05 and z > 0 else neg_color if p < 0.05 and z < 0 else grey_color for (z, p) in zip(z_scores1, p_values1)]
    colors2 = [pos_color if p < 0.05 and z > 0 else neg_color if p < 0.05 and z < 0 else grey_color for (z, p) in zip(z_scores2, p_values2)]
    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(motif_ids_2, z_scores1, c=colors1, s=140, edgecolor='black', linewidth=0.8, alpha=0.9, marker='o', label='Z-Scores bsky (circles)')
    plt.scatter(motif_ids_2, z_scores2, c=colors2, s=140, edgecolor='black', linewidth=0.8, alpha=0.9, marker='s', label='Z-Scores ts (squares)')
    plt.axhline(y=0, color='#222222', linestyle='dashed', linewidth=1.5)
    plt.xlabel('Motif ID', fontsize=14, fontweight='bold')
    plt.ylabel('Z-score', fontsize=14, fontweight='bold')
    plt.title('Comparison of Z-Scores (Sign Color, Shape Difference, Grey for Non-Significant)', fontsize=16, fontweight='bold', color='#333333')
    plt.xticks(motif_ids_2, fontsize=12, color='#333333')
    plt.yticks(fontsize=12, color='#333333')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, edgecolor='black', fontsize=12, loc='upper right')
    plt.show()
    return (
        colors1,
        colors2,
        grey_color,
        motif_ids_2,
        neg_color,
        p_values1,
        p_values2,
        pos_color,
        z_scores1,
        z_scores2,
    )


@app.cell
def _(norm, np, plt):
    motif_ids_3 = np.arange(13)
    z_scores1_1 = np.array([41.4376, -29.1637, -26.3728, -12.347, -33.526, -11.219, 57.8398, -47.8141, 40.985, 9.88934, -38.2146, -5.17001, 48.4784])
    z_scores2_1 = np.array([3.20728, 10.1647, -21.2754, 15.8421, -12.1197, -1.16916, 39.1044, -10.9542, 0.214503, 13.5716, -44.846, -7.57559, 47.2986])
    p_values1_1 = 2 * (1 - norm.cdf(np.abs(z_scores1_1)))
    p_values2_1 = 2 * (1 - norm.cdf(np.abs(z_scores2_1)))
    grey_color_1 = '#d3d3d3'
    neg_color_1 = '#90a0c8'
    pos_color_1 = '#f4c28f'
    colors1_1 = [pos_color_1 if p < 0.05 and z > 0 else neg_color_1 if p < 0.05 and z < 0 else grey_color_1 for (z, p) in zip(z_scores1_1, p_values1_1)]
    colors2_1 = [pos_color_1 if p < 0.05 and z > 0 else neg_color_1 if p < 0.05 and z < 0 else grey_color_1 for (z, p) in zip(z_scores2_1, p_values2_1)]
    plt.figure(figsize=(10, 6), dpi=300)
    for i in range(len(motif_ids_3)):
        plt.plot([motif_ids_3[i], motif_ids_3[i]], [z_scores1_1[i], z_scores2_1[i]], color='gray', linestyle='solid', alpha=0.7)
    plt.scatter(motif_ids_3, z_scores1_1, color=colors1_1, s=140, edgecolor='black', linewidth=0.8, label='Z-Scores bsky', marker='*')
    plt.scatter(motif_ids_3, z_scores2_1, color=colors2_1, s=140, edgecolor='black', linewidth=0.8, label='Z-Scores ts', marker='X')
    plt.axhline(y=0, color='#222222', linestyle='solid', linewidth=1.5)
    plt.xlabel('Motif ID', fontsize=14, fontweight='bold')
    plt.ylabel('Z-score', fontsize=14, fontweight='bold')
    plt.title('Paired Lollipop Plot: Z-Score Differences (bsky vs. ts)', fontsize=16, fontweight='bold', color='#333333')
    plt.xticks(motif_ids_3, fontsize=12, color='#333333')
    plt.yticks(fontsize=12, color='#333333')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(frameon=True, edgecolor='black', fontsize=12, loc='upper right')
    plt.show()
    return (
        colors1_1,
        colors2_1,
        grey_color_1,
        i,
        motif_ids_3,
        neg_color_1,
        p_values1_1,
        p_values2_1,
        pos_color_1,
        z_scores1_1,
        z_scores2_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Calculate statistics""")
    return


@app.cell
def _(cascade_graph):
    stats = cascade_graph.calculate_statistics()
    return (stats,)


@app.cell
def _(pd, stats):
    # build the dataframe, column and row switch

    combined_stats_df = pd.DataFrame(stats["combined_graph"]).T
    repost_stats_df = pd.DataFrame(stats["repost_graph"]).T
    reply_stats_df = pd.DataFrame(stats["reply_graph"]).T
    return combined_stats_df, reply_stats_df, repost_stats_df


@app.cell
def _(pd):
    # import the topic data
    bsky_topics = pd.read_csv("../data/bsky_df_id_topic.csv")
    return (bsky_topics,)


@app.cell
def _(bsky_topics):
    bsky_topics
    return


@app.cell
def _(combined_graph, repost_stats_df):
    # find root id for each repost id


    def find_root(G, child):
        parent = list(G.predecessors(child))
        if len(parent) == 0:
            return child
        else:
            return find_root(G, parent[0])


    for repost_id in repost_stats_df.index:
        if combined_graph.in_degree(repost_id) == 0:
            repost_stats_df.loc[repost_id, "root_id"] = repost_id
        else:
            repost_stats_df.loc[repost_id, "root_id"] = find_root(combined_graph, repost_id)
    return find_root, repost_id


@app.cell
def _(cascade_graph):
    combined_graph_1 = cascade_graph.build_combined_graph()
    return (combined_graph_1,)


@app.cell
def _(repost_stats_df):
    repost_stats_df.reset_index(inplace=True)
    return


@app.cell
def _(bsky_topics, repost_stats_df):
    repost_original = repost_stats_df.merge(
        bsky_topics, left_on="root_id", right_on="id", how="left"
    ).drop(columns="id")
    return (repost_original,)


@app.cell
def _(repost_original):
    repost_original.to_csv("../data/bsky_repost_stat.csv", index=False)
    return


@app.cell
def _(bsky_topics, reply_stats_df):
    reply_stats_df.reset_index(inplace=True)
    reply_original = reply_stats_df.merge(
        bsky_topics, left_on="index", right_on="id", how="left"
    ).drop(columns="id")
    return (reply_original,)


@app.cell
def _(reply_original):
    reply_original.to_csv("../data/bsky_reply_stats.csv", index=False)
    return


@app.cell
def _(bsky_topics, combined_stats_df):
    combined_stats_df.reset_index(inplace=True)
    combined_original = combined_stats_df.merge(
        bsky_topics, left_on="index", right_on="id", how="left"
    ).drop(columns="id")

    combined_original.to_csv("../data/bsky_combined_stats.csv", index=False)
    return (combined_original,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Power law distribution""")
    return


@app.cell
def _(pd):
    repost_original_1 = pd.read_csv('../data/bsky_repost_stat.csv')
    reply_original_1 = pd.read_csv('../data/bsky_reply_stats.csv')
    combined_original_1 = pd.read_csv('../data/bsky_combined_stats.csv')
    return combined_original_1, reply_original_1, repost_original_1


@app.cell
def _(json):
    with open('../data/bsky_author_info.json') as f_1:
        bsky_author_info = json.load(f_1)
    return bsky_author_info, f_1


@app.cell
def _(bsky_author_info, bsky_repost):
    author_analysis = []
    from tqdm.auto import tqdm
    for post_1 in tqdm(bsky_repost):
        post_id_1 = post_1['_id']
        repost_author = post_1['author']['did']
        repost_author_name = post_1['author'].get('displayName', 'nan')
        repost_author_info = bsky_author_info.get(repost_author, {}).get('followersCount', 0)
        post_1['record']['follower_count'] = repost_author_info
        author_analysis.append({'post_id': post_id_1, 'repost_author': repost_author, 'repost_author_name': repost_author_name, 'follower_count': repost_author_info})
    return (
        author_analysis,
        post_1,
        post_id_1,
        repost_author,
        repost_author_info,
        repost_author_name,
        tqdm,
    )


@app.cell
def _(author_analysis, pd):
    df_author_analysis = pd.DataFrame(author_analysis)
    return (df_author_analysis,)


@app.cell
def _(df_author_analysis, repost_original_1):
    df_author_analysis_1 = df_author_analysis.merge(repost_original_1, left_on='post_id', right_on='index', how='left')
    return (df_author_analysis_1,)


@app.cell
def _(df_author_analysis_1):
    df_author_analysis_1
    return


@app.cell
def _(df_author_analysis_1, np, plt):
    import powerlaw
    follower = df_author_analysis_1.groupby('repost_author')['follower_count'].mean().values
    degree_sequence = np.array(follower) + 1
    fit = powerlaw.Fit(degree_sequence)
    print(f'Power-law exponent (gamma): {fit.alpha}')
    print(f'Xmin: {fit.xmin}')
    (R, p) = fit.distribution_compare('power_law', 'exponential')
    print(f'Power-law vs. Exponential: R={R}, p={p}')
    powerlaw.plot_pdf(degree_sequence, color='b', label='Empirical Data')
    fit.power_law.plot_pdf(color='r', linestyle='--', label='Power-law Fit')
    plt.legend()
    plt.show()
    return R, degree_sequence, fit, follower, p, powerlaw


@app.cell
def _(df_author_analysis_1, np, plt, powerlaw):
    follower_1 = df_author_analysis_1.groupby('repost_author')['size'].mean().values
    degree_sequence_1 = np.array(follower_1) + 1
    fit_1 = powerlaw.Fit(degree_sequence_1)
    print(f'Power-law exponent (gamma): {fit_1.alpha}')
    print(f'Xmin: {fit_1.xmin}')
    (R_1, p_1) = fit_1.distribution_compare('power_law', 'exponential')
    print(f'Power-law vs. Exponential: R={R_1}, p={p_1}')
    powerlaw.plot_pdf(degree_sequence_1, color='b', label='Empirical Data')
    fit_1.power_law.plot_pdf(color='r', linestyle='--', label='Power-law Fit')
    plt.legend()
    plt.show()
    return R_1, degree_sequence_1, fit_1, follower_1, p_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Truthsocial""")
    return


@app.cell
def _(json):
    with open('../data/ts_threads_withReblogs.json') as f_2:
        ts_repost = json.load(f_2)
    with open('../data/ts_user_following_map.json') as f_2:
        ts_follow = json.load(f_2)
    with open('../data/ts_post_to_label.json') as f_2:
        ts_post_to_label = json.load(f_2)
    return f_2, ts_follow, ts_post_to_label, ts_repost


@app.cell
def _(ts_repost):
    ts_repost[1]
    return


@app.cell
def _(ts_post_to_label, ts_repost):
    ts_ideology_map = {}
    import re
    from bs4 import BeautifulSoup
    for post_2 in ts_repost:
        post_id_2 = post_2['_id']
        content = post_2['content']
        soup = BeautifulSoup(content, 'html.parser')
        for a_tag in soup.find_all('a'):
            a_tag.unwrap()
        plain_text = soup.get_text(separator=' ')
        text_1 = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F])|\\s)+', '', plain_text)
        ideology_1 = ts_post_to_label.get(text_1, 'center')
        if ideology_1 == 'lean left':
            ideology_1 = 'left'
        elif ideology_1 == 'lean right':
            ideology_1 = 'right'
        ts_ideology_map[post_id_2] = ideology_1
    return (
        BeautifulSoup,
        a_tag,
        content,
        ideology_1,
        plain_text,
        post_2,
        post_id_2,
        re,
        soup,
        text_1,
        ts_ideology_map,
    )


@app.cell
def _(cascade_analysis, ts_follow, ts_ideology_map, ts_repost):
    import importlib

    importlib.reload(cascade_analysis)

    ts_cascade_graph = cascade_analysis.InformationCascadeGraph(ts_repost, ts_follow,ts_ideology_map)
    return importlib, ts_cascade_graph


@app.cell
def _(ts_cascade_graph):
    ts_reply_graph = ts_cascade_graph.build_reply_graph()
    return (ts_reply_graph,)


@app.cell
def _(ts_cascade_graph):
    ts_repost_graph = ts_cascade_graph.build_repost_graph()
    return (ts_repost_graph,)


@app.cell
def _(cp):
    import gc

    gc.collect()  # Garbage collection for CPU
    cp.get_default_memory_pool().free_all_blocks()  # Free all GPU memory
    cp.get_default_pinned_memory_pool().free_all_blocks()  # Free pinned memory
    return (gc,)


@app.cell
def _(cascade_graph):
    combined_graph_2 = cascade_graph.build_combined_graph()
    return (combined_graph_2,)


@app.cell
def _(cascade_graph, combined_graph_2, reply_graph, repost_graph):
    reply_stats = cascade_graph.calculate_tree_statistics(reply_graph)
    repost_stats = cascade_graph.calculate_tree_statistics(repost_graph)
    combined_stats = cascade_graph.calculate_tree_statistics(combined_graph_2)
    return combined_stats, reply_stats, repost_stats


@app.cell
def _(combined_stats, pd, reply_stats, repost_stats):
    reply_stats_df_1 = pd.DataFrame(reply_stats).T
    repost_stats_df_1 = pd.DataFrame(repost_stats).T
    combined_stats_df_1 = pd.DataFrame(combined_stats).T
    return combined_stats_df_1, reply_stats_df_1, repost_stats_df_1


@app.cell
def _(combined_graph_2, find_root, repost_stats_df_1):
    repost_root = []

    def find_root_1(G, child):
        parent = list(G.predecessors(child))
        if len(parent) == 0:
            return child
        else:
            return find_root(G, parent[0])
    for repost_id_1 in repost_stats_df_1.index:
        if combined_graph_2.in_degree(repost_id_1) == 0:
            repost_stats_df_1.loc[repost_id_1, 'root_id'] = repost_id_1
        else:
            repost_stats_df_1.loc[repost_id_1, 'root_id'] = find_root_1(combined_graph_2, repost_id_1)
    return find_root_1, repost_id_1, repost_root


@app.cell
def _(pd):
    ts_topics = pd.read_csv("../data/ts_df_id_topic.csv")
    return (ts_topics,)


@app.cell
def _(reply_stats_df_1, ts_topics):
    reply_stats_df_1.reset_index(inplace=True)
    reply_stats_df_1['index'] = reply_stats_df_1['index'].astype(int)
    ts_topics['id'] = ts_topics['id'].astype(int)
    reply_stats_df_2 = reply_stats_df_1.merge(ts_topics, left_on='index', right_on='id', how='left').drop(columns='id')
    return (reply_stats_df_2,)


@app.cell
def _(reply_stats_df_2):
    reply_stats_df_2.to_csv('../data/ts_reply_stats.csv', index=False)
    return


@app.cell
def _(repost_stats_df_1, ts_topics):
    repost_stats_df_1.reset_index(inplace=True)
    repost_stats_df_1['root_id'] = repost_stats_df_1['root_id'].astype(int)
    repost_stats_df_1['index'] = repost_stats_df_1['index'].astype(int)
    ts_topics['id'] = ts_topics['id'].astype(int)
    repost_stats_df_test = repost_stats_df_1.merge(ts_topics, left_on='root_id', right_on='id', how='left').drop(columns='id')
    return (repost_stats_df_test,)


@app.cell
def _(repost_stats_df_test):
    repost_stats_df_test.to_csv("../data/ts_repost_stats.csv", index=False)
    return


@app.cell
def _(combined_stats_df_1, ts_topics):
    combined_stats_df_1.reset_index(inplace=True)
    combined_stats_df_1['index'] = combined_stats_df_1['index'].astype(int)
    ts_topics['id'] = ts_topics['id'].astype(int)
    combined_stats_df_output = combined_stats_df_1.merge(ts_topics, left_on='index', right_on='id', how='left').drop(columns='id')
    return (combined_stats_df_output,)


@app.cell
def _(combined_stats_df_output):
    combined_stats_df_output.to_csv("../data/ts_combined_stats.csv", index=False)
    return


@app.cell
def _(pd):
    bsky_df = pd.read_csv("../data/bsky_reply_stats.csv")
    return (bsky_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Powerlaw analysis""")
    return


@app.cell
def _(pd):
    ts_repost_df = pd.read_csv("../data/ts_repost_stats.csv")
    ts_reply_df = pd.read_csv("../data/ts_reply_stats.csv")
    return ts_reply_df, ts_repost_df


@app.cell
def _(ts_repost):
    ts_author_analysis = []
    for post_3 in ts_repost:
        post_id_3 = post_3['_id']
        repost_author_1 = post_3['account']['id']
        repost_author_name_1 = post_3['account'].get('username', 'nan')
        repost_author_info_1 = post_3['account'].get('followers_count', 0)
        ts_author_analysis.append({'post_id': post_id_3, 'repost_author': repost_author_1, 'repost_author_name': repost_author_name_1, 'follower_count': repost_author_info_1})
    return (
        post_3,
        post_id_3,
        repost_author_1,
        repost_author_info_1,
        repost_author_name_1,
        ts_author_analysis,
    )


@app.cell
def _(pd, ts_author_analysis):
    ts_author_analysis_df = pd.DataFrame(ts_author_analysis)
    return (ts_author_analysis_df,)


@app.cell
def _(ts_repost_df):
    ts_repost_df['index'] = ts_repost_df['index'].astype(str)
    return


@app.cell
def _(ts_author_analysis_df, ts_repost_df):
    ts_author_analysis_df_1 = ts_author_analysis_df.merge(ts_repost_df, left_on='post_id', right_on='index', how='left')
    return (ts_author_analysis_df_1,)


@app.cell
def _(ts_author_analysis_df_1):
    ts_author_analysis_df_1.groupby('repost_author')[['follower_count', 'size']].mean()
    return


@app.cell
def _(ts_author_analysis_df_1):
    ts_author_analysis_df_1.groupby('repost_author_name')['follower_count'].mean().sort_values(ascending=False)
    return


@app.cell
def _(np, plt, powerlaw, ts_author_analysis_df_1):
    follower_2 = ts_author_analysis_df_1.groupby('repost_author_name')['follower_count'].mean()
    degree_sequence_2 = np.array(follower_2) + 1
    fit_2 = powerlaw.Fit(degree_sequence_2)
    print(f'Power-law exponent (gamma): {fit_2.alpha}')
    print(f'Xmin: {fit_2.xmin}')
    (R_2, p_2) = fit_2.distribution_compare('power_law', 'exponential')
    print(f'Power-law vs. Exponential: R={R_2}, p={p_2}')
    powerlaw.plot_pdf(degree_sequence_2, color='b', label='Empirical Data')
    fit_2.power_law.plot_pdf(color='r', linestyle='--', label='Power-law Fit')
    plt.legend()
    plt.show()
    return R_2, degree_sequence_2, fit_2, follower_2, p_2


@app.cell
def _(np, plt, powerlaw, ts_author_analysis_df_1):
    follower_3 = ts_author_analysis_df_1.groupby('repost_author_name')['size'].mean()
    degree_sequence_3 = np.array(follower_3) + 1
    fit_3 = powerlaw.Fit(degree_sequence_3)
    print(f'Power-law exponent (gamma): {fit_3.alpha}')
    print(f'Xmin: {fit_3.xmin}')
    (R_3, p_3) = fit_3.distribution_compare('power_law', 'exponential')
    print(f'Power-law vs. Exponential: R={R_3}, p={p_3}')
    powerlaw.plot_pdf(degree_sequence_3, color='b', label='Empirical Data')
    fit_3.power_law.plot_pdf(color='r', linestyle='--', label='Power-law Fit')
    plt.legend()
    plt.show()
    return R_3, degree_sequence_3, fit_3, follower_3, p_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Fix data""")
    return


@app.cell
def _():
    import requests
    url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread"
    headers = {
        "Content-Type": "application/json",
        "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    payload = {
        "uri": "at://maoliumich.bsky.social/app.bsky.feed.post/3lixtxo6gpk2z" ,
        "depth": 99,

    }

    response = requests.get(url, headers=headers,params=payload)
    return headers, payload, requests, response, url


@app.cell
def _(headers, requests, tqdm, uris, url):
    def max_depth(reply):
        global end_reply
        if not reply or 'replies' not in reply or (not reply['replies']):
            if reply['post']['replyCount'] > 0:
                end_reply.append(reply['post']['uri'])
            return 1
        return 1 + max((max_depth(child) for child in reply['replies']))
    end_reply = []
    for (i_1, uri) in tqdm(enumerate(uris), total=len(uris)):
        payload_1 = {'uri': uri, 'depth': 99}
        try:
            response_1 = requests.get(url, headers=headers, params=payload_1)
            response_1.raise_for_status()
        except:
            print('Error: ', uri)
            continue
        response_json = response_1.json()
        depth = max_depth(response_json['thread'])
    return (
        depth,
        end_reply,
        i_1,
        max_depth,
        payload_1,
        response_1,
        response_json,
        uri,
    )


@app.cell
def _(end_reply, headers, max_depth, requests, tqdm, url):
    end_replies_old = end_reply
    all_posts = []
    end_reply_1 = []

    def max_depth_1(reply):
        global end_reply
        if not reply or 'replies' not in reply or (not reply['replies']):
            if reply['post']['replyCount'] > 0:
                end_reply_1.append(reply['post']['uri'])
            return 1
        return 1 + max((max_depth(child) for child in reply['replies']))

    def flatten_thread(thread):
        for reply in thread.get('replies', []):
            all_posts.append(reply)
            flatten_thread(reply)
    for reply_uri in tqdm(end_replies_old, total=len(end_replies_old)):
        uri_1 = reply_uri
        payload_2 = {'uri': uri_1, 'depth': 99}
        try:
            response_2 = requests.get(url, headers=headers, params=payload_2)
            response_2.raise_for_status()
        except:
            print('Error: ', uri_1)
            continue
        response_json_1 = response_2.json()
        flatten_thread(response_json_1['thread'])
        depth_1 = max_depth_1(response_json_1['thread'])
        all_posts.append(response_json_1['thread'])
    return (
        all_posts,
        depth_1,
        end_replies_old,
        end_reply_1,
        flatten_thread,
        max_depth_1,
        payload_2,
        reply_uri,
        response_2,
        response_json_1,
        uri_1,
    )


@app.cell
def _(end_reply_1):
    len(end_reply_1)
    return


@app.cell
def _(end_replies_old, end_reply_1):
    end_replies_old_1 = list(set(end_reply_1) - set(end_replies_old))
    return (end_replies_old_1,)


@app.cell
def _(all_posts):
    len(all_posts)
    return


@app.cell
def _(all_posts):
    all_posts_only_post = [post['post'] for post in all_posts]
    new_posts = []
    for post_4 in all_posts_only_post:
        post_4['_id'] = post_4['uri']
        new_posts.append(post_4)
    return all_posts_only_post, new_posts, post_4


@app.cell
def _(bsky_repost, new_posts):
    bsky_repost.extend(new_posts)
    return


@app.cell
def _(bsky_repost):
    len(bsky_repost)
    return


@app.cell
def _(end_replies_old_1, end_reply_1):
    set(end_replies_old_1) - set(end_reply_1)
    return


@app.cell
def _(end_reply, max_depth_1, response_2):
    response_json_2 = response_2.json()
    end_reply_2 = []

    def max_depth_2(reply):
        global end_reply
        if not reply or 'replies' not in reply or (not reply['replies']):
            if reply['post']['replyCount'] > 0:
                end_reply_2.append(reply['post']['uri'])
            return 1
        return 1 + max((max_depth_1(child) for child in reply['replies']))
    depth_2 = max_depth_2(response_json_2['thread'])
    return depth_2, end_reply_2, max_depth_2, response_json_2


@app.cell
def _(end_reply_2):
    end_reply_2
    return


@app.cell
def _(requests):
    session = requests.Session()
    url_1 = f'https://bsky.social/xrpc/com.atproto.server.createSession'
    payload_3 = {'identifier': 'maoliumich.bsky.social', 'password': '6k8XKfsXmPAvV9q'}
    response_3 = session.post(url_1, json=payload_3)
    return payload_3, response_3, session, url_1


@app.cell
def _(response_3):
    token = response_3.json()['accessJwt']
    return (token,)


@app.cell
def _(token):
    token
    return


@app.cell
def _(response_3):
    response_json_3 = response_3.json()
    return (response_json_3,)


@app.cell
def _(response_json_3):
    missing = response_json_3['thread']['post']
    return (missing,)


@app.cell
def _(missing):
    missing['_id'] = missing['uri']
    return


@app.cell
def _(all_posts):
    all_posts_only_post_1 = [post['post'] for post in all_posts]
    return (all_posts_only_post_1,)


@app.cell
def _(all_posts_only_post_1):
    len(all_posts_only_post_1)
    return


@app.cell
def _(all_posts_only_post_1, bsky_repost):
    bsky_repost.extend(all_posts_only_post_1)
    len(bsky_repost)
    return


@app.cell
def _(bsky_repost, json):
    file_to_modify = '../data/bsky_reposts_new.json'
    import shutil
    shutil.copy(file_to_modify, '../data/bsky_reposts_new_copy.json')
    with open('../data/bsky_reposts_new.json', 'w') as f_3:
        json.dump(bsky_repost, f_3)
    return f_3, file_to_modify, shutil


@app.cell
def _(json):
    with open('../data/bsky_reposts_new.json', 'r') as f_4:
        bsky_repost_1 = json.load(f_4)
    return bsky_repost_1, f_4


@app.cell
def _(bsky_repost_1, pd):
    bsky_repost_df = pd.DataFrame(bsky_repost_1)
    return (bsky_repost_df,)


@app.cell
def _(bsky_repost_1):
    bsky_repost_uri = [post['uri'] for post in bsky_repost_1]
    return (bsky_repost_uri,)


@app.cell
def _(bsky_repost_uri):
    print(len(bsky_repost_uri))
    len(set(bsky_repost_uri))
    return


@app.cell
def _(bsky_repost_1, requests, tqdm):
    url_2 = 'https://public.api.bsky.app/xrpc/app.bsky.feed.getRepostedBy'
    count = {}
    headers_1 = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    payload_4 = {'uri': 'at://maoliumich.bsky.social/app.bsky.feed.post/3lixtxo6gpk2z'}
    new_bsky_repost = []
    count = 0
    for post_5 in tqdm(bsky_repost_1, total=len(bsky_repost_1)):
        if post_5.get('reposts', None):
            new_bsky_repost.append(post_5)
            continue
        sum = 0
        while count < post_5['repostCount']:
            uri_2 = post_5['uri']
            payload_4 = {'uri': uri_2, 'limit': 100}
            try:
                response_4 = requests.get(url_2, headers=headers_1, params=payload_4)
                response_4.raise_for_status()
            except:
                print('Error: ', uri_2)
                break
            response_json_4 = response_4.json()
            count = count + len(response_json_4['repostedBy'])
            post_5['reposts'] = response_json_4['repostedBy']
            new_bsky_repost.append(post_5)
            break
        if post_5['repostCount'] > 100:
            count = count + 1
    return (
        count,
        headers_1,
        new_bsky_repost,
        payload_4,
        post_5,
        response_4,
        response_json_4,
        sum,
        uri_2,
        url_2,
    )


@app.cell
def _(bsky_repost_1):
    author_ids = set([post['author']['did'] for post in bsky_repost_1])
    len(author_ids)
    return (author_ids,)


@app.cell
def _(new_bsky_repost):
    count_1 = 0
    for i_2 in new_bsky_repost:
        if len(i_2.get('reposts', [])) > 0:
            count_1 = count_1 + 1
    return count_1, i_2


@app.cell
def _(new_bsky_repost):
    new_bsky_repost
    return


@app.cell
def _(json, new_bsky_repost):
    with open('../data/bsky_reposts_newReposts.json', 'w') as f_5:
        json.dump(new_bsky_repost, f_5)
    return (f_5,)


@app.cell
def _(merged_dict):
    old_authors = set(merged_dict.keys())
    return (old_authors,)


@app.cell
def _(author_ids, old_authors):
    author_to_fetch = list(author_ids - old_authors)
    return (author_to_fetch,)


@app.cell
def _(author_to_fetch, requests, tqdm):
    def fetch_flower_information(base_url, author, limit=100):
        """
        Fetches flower information from the API using pagination.

        Parameters:
            base_url (str): The API endpoint.
            author (str): The author ID.
            limit (int): Number of items to fetch per request.

        Returns:
            list: A list of all flower information retrieved from the API.
        """
        all_flower_info = []
        params = {
            "actor": author,
            "limit": limit
        }

        while True:
            response = requests.get(base_url, params=params)

            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                break

            data = response.json()

            # Add the flower information from the current response
            all_flower_info.extend(data.get("follows", []))
            #all_flower_info.extend(data.get("followers", []))

            # Check if there's a cursor for the next page
            cursor = data.get("cursor")
            if not cursor:
                break

            # Update the params with the new cursor
            params["cursor"] = cursor

        return all_flower_info
    all_flower_info = []
    base_url = "https://public.api.bsky.app/xrpc/app.bsky.graph.getFollows"
    for author in tqdm(author_to_fetch):
        all_flower_info.append({author: fetch_flower_information(base_url, author)})
    return all_flower_info, author, base_url, fetch_flower_information


@app.cell
def _(all_flower_info):
    len(all_flower_info)
    return


@app.cell
def _(all_flower_info, bsky_follow):
    bsky_follow.extend(all_flower_info)
    return


@app.cell
def _(bsky_follow):
    len(bsky_follow)
    return


@app.cell
def _(bsky_follow, json, shutil):
    file_to_modeify = '../data/bsky_follows.json'
    shutil.copy(file_to_modeify, '../data/bsky_follows_copy.json')
    with open(file_to_modeify, 'w') as f_6:
        json.dump(bsky_follow, f_6)
    return f_6, file_to_modeify


@app.cell
def _(requests):
    def get_userinfo(user_id):
        url = f"https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile"
        params = {
            'actor': user_id
        }
        response = requests.get(url, params=params)
        return response.json()
    get_userinfo("did:plc:4qqizocrnriintskkh6trnzv")
    return (get_userinfo,)


app._unparsable_cell(
    r"""
    posts = []
    for post in bsky_repost:
        posts.append((post['_id'], post['record']['text'], post['record'][]))
    """,
    name="_"
)


@app.cell
def _(posts):
    len(set(posts))
    return


@app.cell
def _(json):
    with open('../data/bsky_post_to_label.json', 'r') as f_7:
        bsky_post_to_label_1 = json.load(f_7)
    return bsky_post_to_label_1, f_7


@app.cell
def _(bsky_post_to_label_1):
    len(bsky_post_to_label_1)
    return


@app.cell
def _(bsky_post_to_label_1, posts):
    posts_for_label = list(set(posts) - set(bsky_post_to_label_1.keys()))
    return (posts_for_label,)


@app.cell
def _(posts_for_label):
    posts_for_label
    return


@app.cell
def _(count_1):
    count_1
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
