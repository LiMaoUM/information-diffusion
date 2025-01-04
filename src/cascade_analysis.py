import networkx as nx
from collections import defaultdict
from tqdm.auto import tqdm

class InformationCascadeGraph:
    def __init__(self, post_data, follow_data):
        self.post_data = post_data
        self.follow_data = follow_data
        self.reply_graph = nx.DiGraph()
        self.repost_graph = nx.DiGraph()
        self.combined_graph = nx.DiGraph()

        # Detect necessary fields
        self.author_id_field = self.detect_field(post_data, ['author_id', ('author', 'did')])
        self.user_id_field = self.detect_field(post_data, [ 'did', ('author', 'did')])
        self.in_reply_to_field = self.detect_field(post_data, ['in_reply_to_id', ("record",'reply','parent', 'uri')])

        # Post ID is fixed as '_id'
        self.post_id_field = '_id'

    def detect_field(self, data, possible_fields):
        """
        Detects the correct field name or path from a list of possible fields in the dataset.
        :param data: The dataset to inspect (list of dictionaries).
        :param possible_fields: List of field names or paths to check.
        :return: The detected field name or path.
        """
        for field in possible_fields:
            for entry in data:
                try:
                    if isinstance(field, tuple):
                        value = entry
                        for subkey in field:
                            value = value[subkey]
                        return field
                    elif field in entry:
                        return field
                except (KeyError, TypeError):
                    continue
        raise KeyError(f"None of the fields {possible_fields} were found in the dataset.")

    def build_reply_graph(self):
        self.reply_graph.clear()
        post_dict = {post[self.post_id_field]: post for post in self.post_data}

        for post in tqdm(self.post_data, desc="Building Reply Graph"):
            in_reply_to_id = post.get(self.in_reply_to_field)
            if in_reply_to_id:
                parent_id = in_reply_to_id
                self.reply_graph.add_edge(parent_id, post[self.post_id_field], type='reply')

                # Add metadata
                self.reply_graph.nodes[post[self.post_id_field]].update(post)
                self.reply_graph.nodes[parent_id].update(post_dict.get(parent_id, {}))

        return self.reply_graph

    def build_repost_graph(self):
        self.repost_graph.clear()
        post_dict = {post[self.post_id_field]: post for post in self.post_data}

        for post in tqdm(self.post_data, desc="Building Repost Graph"):
            original_post_id = post[self.post_id_field]
            original_author_id = post.get('author').get('did')
            reposts = post.get('reposts', [])
            all_reposts_users = [repost.get('did') for repost in reposts]

            # Initialize linked and unlinked nodes
            linked_users = {original_author_id: original_post_id} # {author_id: repost_id}
            unlinked_users = {}
            unlinked_nodes = []

            # Assign unique repost IDs and check direct links
            for i, repost in enumerate(reposts):
                repost_author = repost.get('did')
                repost_id = f"{original_post_id}_repost_{i}"

                if original_author_id in self.follow_data.get(repost_author, []):
                    # Directly link to the original author
                    self.repost_graph.add_edge(original_post_id, repost_id, type='repost')
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
                        #check to aviod cycle, i.e., the other way around is already linked
                        if (node_id in self.repost_graph.nodes and nx.has_path(self.repost_graph, node_id, linked_users.get(linked_user))):
                            continue
                        # Link to the first user who follows the node
                        if linked_user in set(linked_users.keys()):
                            self.repost_graph.add_edge(linked_users.get(linked_user), node_id, type='repost')
                        else:
                            self.repost_graph.add_edge(unlinked_users.get(linked_user), node_id, type='repost')
                        linked_users[linked_user] = node_id
                        unlinked_nodes.remove((node, node_id))
                        break
            # Fallback: Link remaining unlinked nodes directly to the original post
            for node, node_id in unlinked_nodes:
                self.repost_graph.add_edge(original_post_id, node_id, type='repost')


        return self.repost_graph




    def build_combined_graph(self):
        self.combined_graph.clear()

        # Start with all nodes and edges from the reply graph
        self.combined_graph.add_edges_from(self.reply_graph.edges(data=True))
        self.combined_graph.add_nodes_from(self.reply_graph.nodes(data=True))

        for reply_node in tqdm(list(self.reply_graph.nodes), desc="Combining Graphs - Reply Nodes"):
            in_edges = list(self.reply_graph.in_edges(reply_node, data=True))

            if in_edges:
                parent_node = in_edges[0][0]
                reply_user = self.reply_graph.nodes[reply_node].get(self.author_id_field)
                parent_user = self.reply_graph.nodes[parent_node].get(self.author_id_field)

                if reply_user not in self.follow_data.get(parent_user, []):
                    # Find the first repost user who the reply user follows
                    for repost_source, repost_target in self.repost_graph.edges():
                        repost_user = self.repost_graph.nodes[repost_target].get(self.author_id_field)

                        if reply_user in self.follow_data.get(repost_user, []):
                            # Redirect the reply edge through the repost target
                            self.combined_graph.remove_edge(parent_node, reply_node)
                            self.combined_graph.add_edge(repost_target, reply_node, type='redirected_reply')
                            break

        # Incorporate remaining reposts
        for u, v, data in tqdm(self.repost_graph.edges(data=True), desc="Combining Graphs - Remaining Reposts"):
            if not self.combined_graph.has_edge(u, v):
                self.combined_graph.add_edge(u, v, **data)
                self.combined_graph.nodes[v].update(self.repost_graph.nodes[v])
                self.combined_graph.nodes[u].update(self.repost_graph.nodes[u])

        return self.combined_graph

    def calculate_tree_statistics(self, graph):
        """
        Calculate statistics for each tree (rooted component) in the graph.
        :param graph: A networkx.DiGraph object representing the cascade graph.
        :return: A dictionary where keys are root node IDs and values are statistics dictionaries.
        """
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Graph must be a directed acyclic graph (DAG).")

        # Find all root nodes (nodes with in-degree 0)
        root_nodes = [n for n, d in graph.in_degree() if d == 0]
        tree_statistics = {}

        for root in tqdm(root_nodes, desc="Calculating Tree Statistics"):
            # Extract the subgraph for the current tree
            tree_nodes = nx.descendants(graph, root) | {root}
            tree = graph.subgraph(tree_nodes)

            # Depth: Maximum distance from the root
            depths = nx.single_source_shortest_path_length(tree, root)
            max_depth = max(depths.values())

            # Size: Total number of nodes
            size = tree.number_of_nodes()

            # Breadth: Number of nodes at each level
            breadth = defaultdict(int)
            for depth in depths.values():
                breadth[depth] += 1

            # Structural Virality: Compute the average pairwise shortest path
            total_distance = 0
            pair_count = 0
            for node in tree.nodes:
                distances = nx.single_source_shortest_path_length(tree, node)
                total_distance += sum(distances.values())
                pair_count += len(distances) - 1  # Exclude self-pairs

            structural_virality = (
                total_distance / pair_count if pair_count > 0 else 0
            )

            # Reach: Total unique users affected
            reach = len(tree.nodes)

            tree_statistics[root] = {
                "max_depth": max_depth,
                "size": size,
                "breadth": dict(breadth),
                "structural_virality": structural_virality,
                "reach": reach,
            }

        return tree_statistics


    def calculate_statistics(self):
        """
        Calculate statistics for all graphs (reply, repost, combined).
        :return: A dictionary containing statistics for each graph.
        """
        return {
            "reply_graph": self.calculate_tree_statistics(self.reply_graph),
            "repost_graph": self.calculate_tree_statistics(self.repost_graph),
            "combined_graph": self.calculate_tree_statistics(self.combined_graph)
        }

    def export_graph(self, graph, format='json'):
        if format == 'json':
            return nx.node_link_data(graph)
        elif format == 'gml':
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
