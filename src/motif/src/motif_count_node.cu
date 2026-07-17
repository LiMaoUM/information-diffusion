#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <queue>

#define BLOCK_SIZE 256
#define NUM_NULL_MODELS 100
#define NUM_STAR_MOTIF_TYPES 27
#define NUM_CHAIN_MOTIF_TYPES 27
#define NUM_MOTIF_TYPES (NUM_STAR_MOTIF_TYPES + NUM_CHAIN_MOTIF_TYPES)

struct Edge {
    int src, dst, src_ideology, dst_ideology;
};

std::vector<Edge> load_graph_from_csv(const std::string& filename, int &num_nodes) {
    std::vector<Edge> edges;
    std::ifstream file(filename);
    std::string line;
    int max_node = 0;

    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string src_str, dst_str, src_ideo_str, dst_ideo_str;
        std::getline(ss, src_str, ',');
        std::getline(ss, dst_str, ',');
        std::getline(ss, src_ideo_str, ',');
        std::getline(ss, dst_ideo_str, ',');
        int src = std::stoi(src_str);
        int dst = std::stoi(dst_str);
        int src_ideo = std::stoi(src_ideo_str);
        int dst_ideo = std::stoi(dst_ideo_str);
        edges.push_back({src, dst, src_ideo, dst_ideo});
        max_node = std::max(max_node, std::max(src, dst));
    }
    file.close();
    num_nodes = max_node + 1;
    return edges;
}

std::vector<Edge> generate_null_model(const std::vector<Edge>& edges, int num_nodes) {
    std::vector<std::vector<int>> adj(num_nodes);
    for (const auto& e : edges) {
        adj[e.src].push_back(e.dst);
        adj[e.dst].push_back(e.src); // assuming undirected for component detection
    }

    // 1. Label connected components using BFS
    std::vector<int> component_id(num_nodes, -1);
    int current_component = 0;
    for (int i = 0; i < num_nodes; ++i) {
        if (component_id[i] != -1) continue;
        std::queue<int> q;
        q.push(i);
        component_id[i] = current_component;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                if (component_id[v] == -1) {
                    component_id[v] = current_component;
                    q.push(v);
                }
            }
        }
        current_component++;
    }

    // 2. Group edges by component
    std::unordered_map<int, std::vector<Edge>> component_edges;
    for (const auto& e : edges) {
        int cid = component_id[e.src];
        component_edges[cid].push_back(e);
    }

    std::vector<Edge> null_edges;
    std::random_device rd;
    std::mt19937 gen(rd());

    // 3. Rewire edges within each component
    for (auto& [cid, comp_edges] : component_edges) {

        if (comp_edges.size() < 2) {
            null_edges.insert(null_edges.end(), comp_edges.begin(), comp_edges.end());
            continue; // Skip components with less than 2 nodes
        }
        int m = comp_edges.size();
        std::uniform_int_distribution<> dis(0, m - 1);
        for (int i = 0; i < m * 5; ++i) {
            int a = dis(gen), b = dis(gen);
            if (a == b) continue;
            Edge& e1 = comp_edges[a];
            Edge& e2 = comp_edges[b];
            if (e1.src != e2.src && e1.dst != e2.dst && e1.src != e2.dst && e1.dst != e2.src) {
                std::swap(e1.dst, e2.dst);
                std::swap(e1.dst_ideology, e2.dst_ideology);
            }
        }
        null_edges.insert(null_edges.end(), comp_edges.begin(), comp_edges.end());
    }

    return null_edges;
}

void build_adjacency_list(const std::vector<Edge>& edges, std::vector<int>& adj_list, std::vector<int>& adj_offset, std::vector<int>& ideologies, std::vector<int>& node_ideologies, int num_nodes) {
    adj_offset.assign(num_nodes + 1, 0);
    node_ideologies.assign(num_nodes, 0);
    for (const auto& edge : edges) adj_offset[edge.src + 1]++;
    for (int i = 1; i <= num_nodes; i++) adj_offset[i] += adj_offset[i - 1];
    adj_list.resize(edges.size());
    ideologies.resize(edges.size());
    std::vector<int> temp_offset = adj_offset;
    for (const auto& edge : edges) {
        int pos = temp_offset[edge.src]++;
        adj_list[pos] = edge.dst;
        ideologies[pos] = edge.dst_ideology;
        node_ideologies[edge.src] = edge.src_ideology;
    }
}

__device__ int enumerate_motif(int a, int b, int c) {
    return a * 9 + b * 3 + c;
}

__global__ void classify_subtrees(int *adj_offset, int *adj_list, int *ideologies, int *node_ideologies, int num_nodes, unsigned int *counts) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= num_nodes) return;
    
    int a_ideo = node_ideologies[a];

    int start = adj_offset[a];
    int end = adj_offset[a + 1];

    for (int i = start; i < end; i++) {
        int b = adj_list[i];
        int b_ideo = ideologies[i];
        for (int j = i + 1; j < end; j++) {
            int c_ideo = ideologies[j];
            int motif = enumerate_motif(a_ideo, b_ideo, c_ideo);
            if (motif >= 0 && motif < NUM_STAR_MOTIF_TYPES) atomicAdd(&counts[motif], 1);
        }
        int b_start = adj_offset[b];
        int b_end = adj_offset[b + 1];
        int b_real_ideo = node_ideologies[b];
        for (int j = b_start; j < b_end; j++) {
            int c_ideo = ideologies[j];
            int motif = enumerate_motif(a_ideo, b_real_ideo, c_ideo);
            if (motif >= 0 && motif < NUM_CHAIN_MOTIF_TYPES) atomicAdd(&counts[NUM_STAR_MOTIF_TYPES + motif], 1);
        }
    }
}

void save_motif_counts(const std::string& filename, const thrust::host_vector<unsigned int>& counts) {
    std::ofstream file(filename);
    for (size_t i = 0; i < counts.size(); ++i) {
        file << "Motif " << i << ": " << counts[i] << "\n";
    }
    file.close();
}

void compute_z_scores(const thrust::host_vector<unsigned int>& real_counts, const std::vector<std::vector<unsigned int>>& null_counts, const std::string& output_zscores_file) {
    std::vector<double> null_means(NUM_MOTIF_TYPES, 0), null_stds(NUM_MOTIF_TYPES, 0);
    std::ofstream zscore_file(output_zscores_file);

    for (int i = 0; i < NUM_MOTIF_TYPES; i++) {
        double sum = 0;
        for (int j = 0; j < NUM_NULL_MODELS; j++) sum += null_counts[j][i];
        null_means[i] = sum / NUM_NULL_MODELS;
    }

    for (int i = 0; i < NUM_MOTIF_TYPES; i++) {
        double sum_sq_diff = 0;
        for (int j = 0; j < NUM_NULL_MODELS; j++) {
            double diff = null_counts[j][i] - null_means[i];
            sum_sq_diff += diff * diff;
        }
        double variance = sum_sq_diff / (NUM_NULL_MODELS - 1);
        null_stds[i] = (variance > 0) ? sqrt(variance) : 1e-9;
    }

    for (int i = 0; i < NUM_MOTIF_TYPES; i++) {
        double z_score = (real_counts[i] - null_means[i]) / null_stds[i];
        zscore_file << "Motif " << i << " Real: " << real_counts[i]
            << ", Mean: " << null_means[i]
            << ", Std: " << null_stds[i]
            << ", Z-score: " << z_score << "\n";

    }
    zscore_file.close();
}
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_graph_csv> <output_counts_file> <output_zscores_file>\n";
        return 1;
    }

    std::string input_csv = argv[1];
    std::string output_counts_file = argv[2];
    std::string output_zscores_file = argv[3];

    int num_nodes = 0;
    std::vector<Edge> edges = load_graph_from_csv(input_csv, num_nodes);

    std::vector<int> adj_list, adj_offset, ideologies, node_ideologies;
    build_adjacency_list(edges, adj_list, adj_offset, ideologies, node_ideologies, num_nodes);

    thrust::device_vector<int> d_adj_list = adj_list;
    thrust::device_vector<int> d_adj_offset = adj_offset;
    thrust::device_vector<int> d_ideologies = ideologies;
    thrust::device_vector<int> d_node_ideologies = node_ideologies;
    thrust::device_vector<unsigned int> d_real_counts(NUM_MOTIF_TYPES, 0);

    classify_subtrees<<<(num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_adj_offset.data()),
        thrust::raw_pointer_cast(d_adj_list.data()),
        thrust::raw_pointer_cast(d_ideologies.data()),
        thrust::raw_pointer_cast(d_node_ideologies.data()),
        num_nodes,
        thrust::raw_pointer_cast(d_real_counts.data())
    );
    cudaDeviceSynchronize();

    thrust::host_vector<unsigned int> real_counts = d_real_counts;
    save_motif_counts(output_counts_file, real_counts);

    std::vector<std::vector<unsigned int>> null_counts(NUM_NULL_MODELS, std::vector<unsigned int>(NUM_MOTIF_TYPES, 0));
    for (int i = 0; i < NUM_NULL_MODELS; i++) {
        std::vector<Edge> null_edges = generate_null_model(edges, num_nodes); 
        build_adjacency_list(null_edges, adj_list, adj_offset, ideologies, node_ideologies, num_nodes);

        d_adj_list = adj_list;
        d_adj_offset = adj_offset;
        d_ideologies = ideologies;
        d_node_ideologies = node_ideologies;
        thrust::device_vector<unsigned int> d_null_counts(NUM_MOTIF_TYPES, 0);

        classify_subtrees<<<(num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(d_adj_offset.data()),
            thrust::raw_pointer_cast(d_adj_list.data()),
            thrust::raw_pointer_cast(d_ideologies.data()),
            thrust::raw_pointer_cast(d_node_ideologies.data()),
            num_nodes,
            thrust::raw_pointer_cast(d_null_counts.data())
        );
        cudaDeviceSynchronize();

        thrust::host_vector<unsigned int> h_null_counts = d_null_counts;
        for (int j = 0; j < NUM_MOTIF_TYPES; j++) {
            null_counts[i][j] = h_null_counts[j];
        }
    }

    compute_z_scores(real_counts, null_counts, output_zscores_file);
    return 0;
}
