from collections import defaultdict

import networkx as nx
from tqdm.auto import tqdm
import cudf
import cugraph
import networkx as nx
import pandas as pd
from tqdm.auto import tqdm
from typing import Literal

class InformationCascadeGraph:
    def __init__(self, post_data: list[dict], follow_data: dict, ideology_map: dict, author_ideology_map: dict, author_ideology_threshold = 0.8, platform: str = "ts"):
        self.post_data = post_data
        self.follow_data = follow_data
        self.ideology_map = ideology_map
        self.author_ideology_map = author_ideology_map
        self.author_ideology_threshold = author_ideology_threshold
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

    def build_reply_graph(self, edge_mapping_type: Literal['normal', 'breakdown'] = 'normal'):
        self.reply_graph.clear()
        post_dict = {post[self.post_id_field]: post for post in self.post_data}
        post_to_author = {
            post[self.post_id_field]: self.get_nested_value(post, self.author_id_field)
            for post in self.post_data
        }
        missing_ideology = 0

        for post in tqdm(self.post_data, desc="Building Reply Graph"):
            post_ideology = self.ideology_map.get(post[self.post_id_field], 'center')
            author_ideology_dist = self.author_ideology_map.get(
                self.get_nested_value(post, self.author_id_field), {}
            )
            if author_ideology_dist.get('left') >= self.author_ideology_threshold:
                author_ideology = 'left'
            elif author_ideology_dist.get('right') >= self.author_ideology_threshold:
                author_ideology = 'right'
            else:
                author_ideology = 'center'
                
            self.reply_graph.add_node(
                post[self.post_id_field],
                author_id=self.get_nested_value(post, self.author_id_field),
                ideology=post_ideology,
                author_ideology=author_ideology
            )
            
            in_reply_to_id = self.get_nested_value(post, self.in_reply_to_field)

            if in_reply_to_id:
                parent_id = in_reply_to_id
                parent_author = post_to_author.get(parent_id, None)
                current_author = post_to_author[post[self.post_id_field]]
                parent_author_ideology_dist = self.author_ideology_map.get(
                    parent_author, 
                )
                if parent_author_ideology_dist is None:
                    missing_ideology += 1
                    continue
                if parent_author_ideology_dist.get('left') >= self.author_ideology_threshold:
                    parent_author_ideology = 'left'
                elif parent_author_ideology_dist.get('right') >= self.author_ideology_threshold:
                    parent_author_ideology = 'right'
                else:
                    parent_author_ideology = 'center'

                is_following = parent_author in self.follow_data.get(current_author, [])
                parent_ideology = self.ideology_map.get(parent_id, 'center')
                
                same_ideology = post_ideology == parent_ideology

                edge_mapping = {
                    (True, True): "directedAligned",
                    (True, False): "directedOpposed",
                    (False, True): "indirectedAligned",
                    (False, False): "indirectedOpposed",
                }

                edge_mapping_breakdown = {
                    ("center", "center"): "centerAligned",
                    ("left", "left"): "leftAligned",
                    ("right", "right"): "rightAligned",
                    ("left", "right"): "leftOpposed",
                    ("right", "left"): "rightOpposed",
                }

                if edge_mapping_type == 'breakdown':
                    edge_type = edge_mapping_breakdown.get(
                        (post_ideology, parent_ideology),
                        "opposed"
                    )
                else:
                    edge_type = edge_mapping[(is_following, same_ideology)]

                same_author_ideology = (
                    author_ideology == parent_author_ideology
                )
                edge_label = edge_mapping[(is_following, same_author_ideology)]


                self.reply_graph.add_edge(
                    parent_id, post[self.post_id_field], type=edge_type, label=edge_label
                )

                # Add metadata
                self.reply_graph.nodes[post[self.post_id_field]]["author_id"] = (
                    self.get_nested_value(post, self.author_id_field)
                )
                self.reply_graph.nodes[parent_id]["author_id"] = self.get_nested_value(
                    post_dict.get(parent_id, {}), self.author_id_field
                )

                # add node ideology
                self.reply_graph.nodes[parent_id]["ideology"] = parent_ideology
                self.reply_graph.nodes[parent_id]["author_ideology"] = parent_author_ideology


        print(
            f"Step 1: {missing_ideology} edges skipped due to missing ideology information"
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

    def calculate_tree_statistics(self, graph, is_reply: bool = False):
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
            total_left = 0
            total_right = 0
            error_idology_count = 0
            total_author_left = 0
            total_author_right = 0
            for node in tree.nodes:
                distances = nx.single_source_shortest_path_length(tree, node)
                total_distance += sum(distances.values())
                pair_count += len(distances) - 1
                if is_reply:
                    node_ideology = tree.nodes[node].get("ideology", None)
                    author_ideology = tree.nodes[node].get("author_ideology", None)
                    if node_ideology == "left":
                        total_left += 1
                    elif node_ideology == "right":
                        total_right += 1
                    if node_ideology is None:
                        error_idology_count += 1
                    if author_ideology == "left":
                        total_author_left += 1
                    elif author_ideology == "right":
                        total_author_right += 1

            structural_virality = total_distance / pair_count if pair_count > 0 else 0

            reach = len(tree.nodes)
            # 👉 Count alignment types
            aligned_edges = sum(
                1 for u, v, data in tree.edges(data=True)
                if data.get("type") in {"directedAligned", "indirectedAligned"}
            )
            total_edges = tree.number_of_edges()
            alignment_ratio = aligned_edges / total_edges if total_edges > 0 else 0
            aligned_author_edges = sum(
                1 for u, v, data in tree.edges(data=True)
                if data.get("label") in {"directedAligned", "indirectedAligned"}
            )
            
            author_alignment_ratio = aligned_author_edges / total_edges if total_edges > 0 else 0
            left_ratio = total_left / size if size > 0 else 0
            right_ratio = total_right / size if size > 0 else 0
            author_left_ratio = total_author_left / size if size > 0 else 0
            author_right_ratio = total_author_right / size if size > 0 else 0

            tree_statistics[root] = {
                "max_depth": max_depth,
                "size": size,
                "breadth": max_breadth,
                "structural_virality": structural_virality,
                "reach": reach,
                "alignment_ratio": alignment_ratio,
                "author_alignment_ratio": author_alignment_ratio,
                "left_ratio": left_ratio,
                "right_ratio": right_ratio,
                "author_left_ratio": author_left_ratio,
                "author_right_ratio": author_right_ratio,
                "error_idology_count": error_idology_count,
                
            }

        return tree_statistics



    def calculate_statistics(self):
        return {
            "reply_graph": self.calculate_tree_statistics(self.reply_graph, is_reply=True),
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
