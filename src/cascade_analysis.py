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
        self.author_id_field = self.detect_field(
            post_data, [("account", "id"), ("author", "did")]
        )
        self.user_id_field = self.detect_field(post_data, ["in_reply_to_account_id", ("author", "did")])
        self.in_reply_to_field = self.detect_field(
            post_data, ["in_reply_to_id", ("record", "reply", "parent", "uri")]
        )

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

        for field in possible_fields:
            for entry in data:
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

        for post in tqdm(self.post_data, desc="Building Reply Graph"):
            self.reply_graph.add_node(
                post[self.post_id_field], author_id=self.get_nested_value(post, self.author_id_field)
            )
            in_reply_to_id = self.get_nested_value(post, self.in_reply_to_field)

            if in_reply_to_id:
                parent_id = in_reply_to_id
                self.reply_graph.add_edge(
                    parent_id, post[self.post_id_field], type="reply"
                )

                # Add metadata
                self.reply_graph.nodes[post[self.post_id_field]]["author_id"] = (
                    self.get_nested_value(post, self.author_id_field)
                )
                self.reply_graph.nodes[parent_id]["author_id"] = self.get_nested_value(post_dict.get(
                    parent_id, {}
                ),self.author_id_field)

        return self.reply_graph

    def build_repost_graph(self):
        self.repost_graph.clear()
        post_dict = {post[self.post_id_field]: post for post in self.post_data}

        for post in tqdm(self.post_data, desc="Building Repost Graph"):
            original_post_id = post[self.post_id_field]
            original_author_id = self.get_nested_value(post, self.author_id_field)
            reposts = post.get('reposts', [])
            all_reposts_users = [repost.get('did') for repost in reposts]

            # Initialize linked and unlinked nodes
            linked_users = {original_author_id: original_post_id} # {author_id: repost_id}
            unlinked_users = {}
            unlinked_nodes = []
            self.repost_graph.add_node(
                original_post_id, author_id=original_author_id
            )


            # Assign unique repost IDs and check direct links
            for i, repost in enumerate(reposts):
                repost_author = repost.get('did')
                repost_id = f"{original_post_id}_repost_{i}"

                if original_author_id in self.follow_data.get(repost_author, []):
                    # Directly link to the original author
                    self.repost_graph.add_edge(original_post_id, repost_id, type='repost')
                    self.repost_graph.nodes[repost_id]['link_type'] = 'direct'
                    self.repost_graph.nodes[repost_id]['author_id'] = repost_author
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
                            self.repost_graph.nodes[node_id]['link_type'] = 'direct'
                            self.repost_graph.nodes[node_id]['author_id'] = node
                        else:
                            self.repost_graph.add_edge(unlinked_users.get(linked_user), node_id, type='repost')
                            self.repost_graph.nodes[node_id]['link_type'] = 'direct'
                            self.repost_graph.nodes[node_id]['author_id'] = node
                        linked_users[linked_user] = node_id

                        unlinked_nodes.remove((node, node_id))

                        break
            # Fallback: Link remaining unlinked nodes directly to the original post
            for node, node_id in unlinked_nodes:
                self.repost_graph.add_edge(original_post_id, node_id, type='repost')
                self.repost_graph.nodes[node_id]['link_type'] = 'fallback'


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
        for u, v, data in tqdm(list(self.reply_graph.edges(data=True)), desc="Merging"):
            reply_user = self.reply_graph.nodes[v].get("author_id")
            parent_user = self.reply_graph.nodes[u].get("author_id")
 

            if parent_user not in self.follow_data.get(reply_user, []):
                # Determine whether the parent node is root
                if self.reply_graph.in_degree(u) == 0:
                    count += 1
                # Find all descendants of the original post
                
                # Find out whether the u is in repost graph
                if u not in self.repost_graph.nodes:
                    print("The node is not in repost graph")
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

        print(f"Step 2.1: Merged {count} reply edges into repost edges.")

        # Step 2.2: Merge fallback repost into reply
        for u, v, data in tqdm(list(self.repost_graph.edges(data=True)), desc="Merging"):
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

            tree_statistics[root] = {
                "max_depth": max_depth,
                "size": size,
                "breadth": max_breadth,
                "structural_virality": structural_virality,
                "reach": reach,
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
