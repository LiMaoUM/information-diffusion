from collections import defaultdict

import networkx as nx
from tqdm.auto import tqdm
import cudf
import cugraph
import networkx as nx
import pandas as pd
from tqdm.auto import tqdm


class InformationCascadeGraph:
    def __init__(self, post_data: list[dict], follow_data: dict, ideology_map: dict, platform: str = "ts"):
        self.post_data = post_data
        self.follow_data = follow_data
        self.ideology_map = ideology_map
        self.reply_graph = nx.DiGraph()
        self.repost_graph = nx.DiGraph()
        self.combined_graph = nx.DiGraph()

        # Detect necessary fields
        self.author_id_field = self.detect_field(
            post_data, [("account", "id"), ("author", "did")]
        )
        self.user_id_field = self.detect_field(
            post_data, ["in_reply_to_account_id", ("author", "did")]
        )
        self.in_reply_to_field = self.detect_field(
            post_data, ["in_reply_to_id", ("record", "reply", "parent", "uri")]
        )
        self.repost_field = self.detect_field(post_data, ["reposts", "reblogList"])
        self.repost_author_field = "id" if platform == "ts" else "did"

        # Post ID is fixed as '_id'
        self.post_id_field = "_id"

    @staticmethod
    def get_nested_value(entry, path):
        if isinstance(path, tuple):
            path = list(path)
            try:
                for key in path:
                    entry = entry[key]
                return entry
            except (KeyError, TypeError):
                return None
        else:
            return entry.get(path)

    def detect_field(self, data, possible_fields):
        """
        Detects the correct field name or path from a list of possible fields in the dataset.
        :param data: The dataset to inspect (list of dictionaries).
        :param possible_fields: List of field names or paths to check.
        :return: The detected field name or path.
        """
        for entry in data:
            for field in possible_fields:
                if isinstance(field, tuple):
                    if self.get_nested_value(entry, field) is not None:
                        return field
                elif field in entry:
                    return field
        raise KeyError(
            f"None of the fields {possible_fields} were found in the dataset."
        )

    def build_reply_graph(self):
        self.reply_graph.clear()
        post_dict = {post[self.post_id_field]: post for post in self.post_data}
        post_to_author = {
            post[self.post_id_field]: self.get_nested_value(post, self.author_id_field)
            for post in self.post_data
        }

        for post in tqdm(self.post_data, desc="Building Reply Graph"):
            self.reply_graph.add_node(
                post[self.post_id_field],
                author_id=self.get_nested_value(post, self.author_id_field),
            )
            in_reply_to_id = self.get_nested_value(post, self.in_reply_to_field)

            if in_reply_to_id:
                parent_id = in_reply_to_id
                parent_author = post_to_author.get(parent_id, None)
                current_author = post_to_author[post[self.post_id_field]]

                is_following = parent_author in self.follow_data.get(current_author, [])
                same_ideology = self.ideology_map.get(post[self.post_id_field],'center') == self.ideology_map.get(in_reply_to_id,'center')

                edge_mapping = {
                    (True, True): "directedAligned",
                    (True, False): "directedOpposed",
                    (False, True): "indirectedAligned",
                    (False, False): "indirectedOpposed",
                }

                edge_type = edge_mapping[(is_following, same_ideology)]


                self.reply_graph.add_edge(
                    parent_id, post[self.post_id_field], type=edge_type
                )

                # Add metadata
                self.reply_graph.nodes[post[self.post_id_field]]["author_id"] = (
                    self.get_nested_value(post, self.author_id_field)
                )
                self.reply_graph.nodes[parent_id]["author_id"] = self.get_nested_value(
                    post_dict.get(parent_id, {}), self.author_id_field
                )

        return self.reply_graph

    def build_repost_graph(self):
        self.repost_graph.clear()

        for post in tqdm(self.post_data, desc="Building Repost Graph"):
            original_post_id = post[self.post_id_field]
            original_author_id = self.get_nested_value(post, self.author_id_field)

            reposts = self.get_nested_value(post, self.repost_field)

            # Initialize linked and unlinked nodes
            linked_users = {
                original_author_id: original_post_id
            }  # {author_id: repost_id}
            unlinked_users = {}
            unlinked_nodes = []
            self.repost_graph.add_node(original_post_id, author_id=original_author_id)
            if reposts is None:
                continue
            all_reposts_users = [
                self.get_nested_value(repost, self.repost_author_field)
                for repost in reposts
            ]

            # Assign unique repost IDs and check direct links
            for i, repost in enumerate(reposts):
                repost_author = self.get_nested_value(repost, self.repost_author_field)
                repost_id = f"{original_post_id}_repost_{i}"

                if original_author_id in self.follow_data.get(repost_author, []):
                    # Directly link to the original author
                    self.repost_graph.add_edge(
                        original_post_id, repost_id, type="repost"
                    )
                    self.repost_graph.nodes[repost_id]["link_type"] = "direct"
                    self.repost_graph.nodes[repost_id]["author_id"] = repost_author
                    linked_users[repost_author] = repost_id
                else:
                    # Add to unlinked nodes for further processing
                    unlinked_users[repost_author] = repost_id
                    unlinked_nodes.append((repost_author, repost_id))
            # Iterative linking for unlinked nodes

            for node, node_id in unlinked_nodes:
                for linked_user in all_reposts_users:
                    if linked_user == node:
                        continue
                    if linked_user in self.follow_data.get(node, []):
                        # check to aviod cycle, i.e., the other way around is already linked
                        if node_id in self.repost_graph.nodes and nx.has_path(
                            self.repost_graph, node_id, linked_users.get(linked_user)
                        ):
                            continue
                        # Link to the first user who follows the node
                        if linked_user in set(linked_users.keys()):
                            self.repost_graph.add_edge(
                                linked_users.get(linked_user), node_id, type="repost"
                            )
                            self.repost_graph.nodes[node_id]["link_type"] = "direct"
                            self.repost_graph.nodes[node_id]["author_id"] = node
                        else:
                            self.repost_graph.add_edge(
                                unlinked_users.get(linked_user), node_id, type="repost"
                            )
                            self.repost_graph.nodes[node_id]["link_type"] = "direct"
                            self.repost_graph.nodes[node_id]["author_id"] = node
                        linked_users[linked_user] = node_id

                        unlinked_nodes.remove((node, node_id))

                        break
            # Fallback: Link remaining unlinked nodes directly to the original post
            for node, node_id in unlinked_nodes:
                self.repost_graph.add_edge(original_post_id, node_id, type="repost")
                self.repost_graph.nodes[node_id]["link_type"] = "fallback"

        return self.repost_graph

    def build_combined_graph(self):
        self.combined_graph.clear()

        # Step 1: Add all nodes and edges from reply and repost graphs
        for u, v, data in self.reply_graph.edges(data=True):
            self.combined_graph.add_edge(u, v, **data)
        for u, v, data in self.repost_graph.edges(data=True):
            self.combined_graph.add_edge(u, v, **data)

        for node, attrs in self.reply_graph.nodes(data=True):
            try:
                self.combined_graph.nodes[node].update(attrs)
            except:
                self.combined_graph.add_node(node, **attrs)
        for node, attrs in self.repost_graph.nodes(data=True):
            try:
                self.combined_graph.nodes[node].update(attrs)
            except:
                self.combined_graph.add_node(node, **attrs)

        # Step 2: Perform deliberate merging
        # Step2.1: Merge reply into repost
        count = 0
        total = 0
        num_not_node = 0
        for u, v, data in tqdm(list(self.reply_graph.edges(data=True)), desc="Merging"):
            reply_user = self.reply_graph.nodes[v].get("author_id")
            parent_user = self.reply_graph.nodes[u].get("author_id")
            if self.reply_graph.in_degree(u) == 0:
                total += 1

            if parent_user not in self.follow_data.get(reply_user, []):
                # Determine whether the parent node is root
                if self.reply_graph.in_degree(u) == 0:
                    count += 1
                    continue
                # Find all descendants of the original post

                # Find out whether the u is in repost graph
                if u not in self.repost_graph.nodes:
                    num_not_node += 1
                    continue

                descendants = nx.descendants(self.repost_graph, u)

                for repost_target in descendants:
                    repost_user = self.repost_graph.nodes[repost_target].get(
                        "author_id"
                    )
                    if repost_user in self.follow_data.get(reply_user, []):
                        self.combined_graph.remove_edge(u, v)
                        self.combined_graph.add_edge(repost_target, v, type="reply")
                        break

        print(
            f"Step 2.1: Merged {count} reply edges into repost edges out of {total} total reply edges"
        )
        print(f"Step 2.1: {num_not_node} nodes not in repost graph")

        # Step 2.2: Merge fallback repost into reply
        for u, v, data in tqdm(
            list(self.repost_graph.edges(data=True)), desc="Merging"
        ):
            if data.get("link_type") == "fallback":
                repost_user = self.repost_graph.nodes[v].get("author_id")
                original_post = u
                original_user = self.repost_graph.nodes[u].get("author_id")

                for reply_target in nx.descendants(self.reply_graph, original_post):
                    reply_user = self.reply_graph.nodes[reply_target].get("author_id")
                    if original_user in self.follow_data.get(reply_user, []):
                        self.combined_graph.remove_edge(u, v)
                        self.combined_graph.add_edge(u, reply_target, type="repost")
                        break

        return self.combined_graph

    def calculate_tree_statistics(self, graph):
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Graph must be a directed acyclic graph (DAG).")

        root_nodes = [n for n, d in graph.in_degree() if d == 0]
        tree_statistics = {}

        for root in tqdm(root_nodes, desc="Calculating Tree Statistics"):
            tree_nodes = nx.descendants(graph, root) | {root}
            tree = graph.subgraph(tree_nodes)

            depths = nx.single_source_shortest_path_length(tree, root)
            max_depth = max(depths.values())

            size = tree.number_of_nodes()

            breadth = defaultdict(int)
            for depth in depths.values():
                breadth[depth] += 1
            # Calculate max breadth
            max_breadth = max(breadth.values())

            total_distance = 0
            pair_count = 0
            for node in tree.nodes:
                distances = nx.single_source_shortest_path_length(tree, node)
                total_distance += sum(distances.values())
                pair_count += len(distances) - 1

            structural_virality = total_distance / pair_count if pair_count > 0 else 0

            reach = len(tree.nodes)
            # 👉 Count alignment types
            aligned_edges = sum(
                1 for u, v, data in tree.edges(data=True)
                if data.get("type") in {"directedAligned", "indirectedAligned"}
            )
            total_edges = tree.number_of_edges()
            alignment_ratio = aligned_edges / total_edges if total_edges > 0 else 0

            tree_statistics[root] = {
                "max_depth": max_depth,
                "size": size,
                "breadth": max_breadth,
                "structural_virality": structural_virality,
                "reach": reach,
                "alignment_ratio": alignment_ratio,
            }

        return tree_statistics

    def calculate_tree_statistics_cugraph(nx_graph):
        """
        Given a NetworkX DiGraph (assumed to be a tree or forest),
        convert it to a cuGraph DiGraph and compute per-root tree statistics.

        Statistics computed for each tree (root):
        - max_depth: maximum distance from the root to any node
        - size: number of nodes in the tree
        - breadth: maximum number of nodes at any distance from the root
        - structural_virality: average shortest-path distance among all node pairs in the tree
        - reach: same as size
        """
        # --- Step 1: Convert the NetworkX graph to a cuGraph graph ---
        # Create an edge list from the NetworkX graph. cuGraph requires a DataFrame with
        # source and destination columns.
        df_edges = nx.to_pandas_edgelist(nx_graph)
        # Make sure the edge list uses the expected column names: 'source' and 'target'
        if "source" not in df_edges.columns or "target" not in df_edges.columns:
            raise ValueError("The edge list must have 'source' and 'target' columns.")

        # Convert the Pandas DataFrame to a cuDF DataFrame.
        cudf_edges = cudf.DataFrame.from_pandas(df_edges)

        # Create a cuGraph DiGraph and load the edge list.
        G_cu = cugraph.Graph(directed=True)
        G_cu.from_cudf_edgelist(cudf_edges, source="source", destination="target")

        # --- Step 2: Identify Root Nodes ---
        # In a tree, root nodes have zero in-degree.
        # We can compute in-degrees by grouping on the 'target' column.
        in_degree_df = (
            cudf_edges.groupby("target")
            .agg({"target": "count"})
            .rename(columns={"target": "in_degree"})
        )
        # Get all unique vertices from both source and target columns.
        all_vertices = pd.concat([df_edges["source"], df_edges["target"]]).unique()
        # Identify roots: vertices that never appear as a target.
        in_degree_set = set(in_degree_df["target"].to_pandas())
        roots = [v for v in all_vertices if v not in in_degree_set]

        tree_statistics = {}
        # --- Step 3: For Each Root, Run BFS and Compute Statistics ---
        for root in tqdm(roots, desc="Calculating Tree Statistics (cuGraph)"):
            # Run BFS from the root; cuGraph returns a cuDF DataFrame with columns:
            # 'vertex', 'distance', and 'predecessor'
            bfs_result = cugraph.bfs(G_cu, root)
            # Convert to Pandas DataFrame for easier (CPU-side) aggregation;
            # if your trees are very large you might want to keep computations on the GPU.
            bfs_pdf = bfs_result.to_pandas()

            # max_depth: the maximum distance encountered
            max_depth = int(bfs_pdf["distance"].max())
            # size (and reach): total number of nodes reached by BFS
            size = len(bfs_pdf)
            # breadth: maximum number of nodes found at the same distance from the root
            breadth_series = bfs_pdf.groupby("distance").size()
            max_breadth = int(breadth_series.max())

            # structural_virality: average distance over all node pairs.
            # The following approach runs a BFS from each node in the tree.
            # (Note: if the trees are large, you might want to use an approximate method.)
            total_distance = 0
            pair_count = 0
            for v in bfs_pdf["vertex"]:
                bfs_v = cugraph.bfs(G_cu, v)
                # Convert the result to Pandas for summing.
                distances = bfs_v.to_pandas()["distance"]
                total_distance += distances.sum()
                # Subtract one so that we don’t count the distance from v to itself
                pair_count += len(distances) - 1
            structural_virality = (
                float(total_distance / pair_count) if pair_count > 0 else 0
            )

            tree_statistics[root] = {
                "max_depth": max_depth,
                "size": size,
                "breadth": max_breadth,
                "structural_virality": structural_virality,
                "reach": size,
            }

        return tree_statistics

    def calculate_statistics(self):
        return {
            "reply_graph": self.calculate_tree_statistics(self.reply_graph),
            "repost_graph": self.calculate_tree_statistics(self.repost_graph),
            "combined_graph": self.calculate_tree_statistics(self.combined_graph),
        }

    def export_graph(self, graph, format="json"):
        if format == "json":
            return nx.node_link_data(graph)
        elif format == "gml":
            return nx.generate_gml(graph)
        else:
            raise ValueError("Unsupported format. Use 'json' or 'gml'.")


# Example usage:
# post_data = [...]  # Load post data
# follow_data = {...}  # Load follow data
# graph_builder = InformationCascadeGraph(post_data, follow_data)
# reply_graph = graph_builder.build_reply_graph()
# repost_graph = graph_builder.build_repost_graph()
# combined_graph = graph_builder.build_combined_graph()
# all_stats = graph_builder.calculate_statistics()
# print(all_stats)
