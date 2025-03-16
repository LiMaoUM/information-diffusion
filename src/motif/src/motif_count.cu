#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define BLOCK_SIZE 256
#define NUM_NULL_MODELS 100

struct Edge {
    int src, dst, type;
};


// Function to read edges and determine the number of nodes
std::vector<Edge> load_graph_from_csv(const std::string& filename, int &num_nodes) {
    std::vector<Edge> edges;
    std::ifstream file(filename);
    std::string line;
    int max_node = 0;
    
    std::cout << "Loading graph from " << filename << "...\n";
    
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string src_str, dst_str, type_str;
        
        std::getline(ss, src_str, ',');
        std::getline(ss, dst_str, ',');
        std::getline(ss, type_str, ',');
        
        int src = std::stoi(src_str);
        int dst = std::stoi(dst_str);
        int type = std::stoi(type_str);
        
        edges.push_back({src, dst, type});
        
        max_node = std::max({max_node, src, dst});
    }
    
    file.close();
    num_nodes = max_node + 1;
    std::cout << "Loaded " << edges.size() << " edges and " << num_nodes << " nodes.\n";
    return edges;
}




// Union-Find (Disjoint Set) data structure
class UnionFind {
public:
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]); // Path compression
        return parent[x];
    }

    void unionSets(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY) {
            // Union by rank
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }

    bool connected(int x, int y) {
        return find(x) == find(y);
    }

private:
    std::vector<int> parent;
    std::vector<int> rank;
};

// Function to generate a null model by swapping edges while ensuring valid structure for a **forest**
std::vector<Edge> generate_null_model(const std::vector<Edge>& edges, int num_swaps, int num_nodes) {
    std::vector<Edge> null_edges = edges;
    UnionFind uf(num_nodes); // Create Union-Find for N nodes

    // Build the initial forest structure
    for (const auto& edge : null_edges) {
        uf.unionSets(edge.src, edge.dst);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, null_edges.size() - 1);

    for (int i = 0; i < num_swaps; ++i) {
        int idx1 = dis(gen);
        int idx2 = dis(gen);

        if (idx1 != idx2) {
            int src1 = null_edges[idx1].src;
            int dst1 = null_edges[idx1].dst;
            int type1 = null_edges[idx1].type;

            int src2 = null_edges[idx2].src;
            int dst2 = null_edges[idx2].dst;
            int type2 = null_edges[idx2].type;

            // Ensure no self-loops
            if (src1 != dst2 && src2 != dst1) {
                // Only swap edges **within the same tree**
                if (uf.find(src1) == uf.find(dst1) && uf.find(src2) == uf.find(dst2)) {
                    // Check if the new swap would create a cycle in any tree
                    if (!uf.connected(src1, dst2) && !uf.connected(src2, dst1)) {
                        // Swap edges and update the Union-Find structure
                        std::swap(null_edges[idx1].dst, null_edges[idx2].dst);
                        std::swap(null_edges[idx1].type, null_edges[idx2].type);
                        uf.unionSets(src1, dst2);
                        uf.unionSets(src2, dst1);
                    }
                }
            }
        }
    }

    return null_edges;
}
// Function to write motif counts to a file
void save_motif_counts(const std::string& filename, const thrust::host_vector<unsigned int>& counts) {
    std::ofstream file(filename);
    for (size_t i = 0; i < counts.size(); ++i) {
        file << "Motif " << i << ": " << counts[i] << "\n";
    }
    file.close();
}

// Function to compute Z-scores with a numerically stable two-pass method
void compute_z_scores(const thrust::host_vector<unsigned int>& real_counts, const std::vector<std::vector<unsigned int>>& null_counts, const std::string& output_zscores_file) {
    std::vector<double> null_means(36, 0), null_stds(36, 0);
    std::ofstream zscore_file(output_zscores_file);
    
    if (!zscore_file) {
        std::cerr << "Error opening file for Z-score computation.\n";
        return;
    }
    
    // First pass: Compute mean
    for (int i = 0; i < 36; i++) {
        double sum = 0;
        for (int j = 0; j < NUM_NULL_MODELS; j++) {
            sum += null_counts[j][i];
        }
        null_means[i] = sum / NUM_NULL_MODELS;
    }

    
    // Second pass: Compute standard deviation
    for (int i = 0; i < 36; i++) {
        double sum_sq_diff = 0;
        for (int j = 0; j < NUM_NULL_MODELS; j++) {
            double diff = null_counts[j][i] - null_means[i];
            sum_sq_diff += diff * diff;
        }
        double variance = sum_sq_diff / (NUM_NULL_MODELS - 1); // Unbiased estimator
        null_stds[i] = (variance > 0) ? sqrt(variance) : 1e-9;
    }

    
    // Compute Z-scores
    for (int i = 0; i < 36; i++) {
        double z_score = (real_counts[i] - null_means[i]) / null_stds[i];
        zscore_file << "Motif " << i << " Z-score: " << z_score << "\n";
    }
    
    zscore_file.close();
}

// Function to build adjacency list
void build_adjacency_list(const std::vector<Edge>& edges, std::vector<int>& adj_list, std::vector<int>& adj_offset, std::vector<int>& edge_types, int num_nodes) {
    adj_offset.assign(num_nodes + 1, 0);
    for (const auto& edge : edges) {
        adj_offset[edge.src + 1]++;
    }
    for (int i = 1; i <= num_nodes; i++) {
        adj_offset[i] += adj_offset[i - 1];
    }
    adj_list.resize(edges.size());
    edge_types.resize(edges.size());
    std::vector<int> temp_offset = adj_offset;
    for (const auto& edge : edges) {
        int pos = temp_offset[edge.src]++;
        adj_list[pos] = edge.dst;
        edge_types[pos] = edge.type;
    }
}

// CUDA Kernel for extracting and classifying 3-node subtrees
__global__ void classify_subtrees(int *adj_offset, int *adj_list, int *edge_types, int num_nodes,  unsigned int *counts) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= num_nodes) return;

    int start_idx = adj_offset[a];
    int end_idx = adj_offset[a + 1];
    if (start_idx >= end_idx) return;

    // Process all 3-node combinations
    for (int i = start_idx; i < end_idx; i++) {
        int b = adj_list[i];
        int edge1_type = edge_types[i];
        if (b < 0 || b >= num_nodes) continue;

        for (int j = i + 1; j < end_idx; j++) {
            int c = adj_list[j];
            int edge2_type = edge_types[j];
            if (c < 0 || c >= num_nodes) continue;

            // Star Motif Mapping
            int motif_type_star = -1;
            if (edge1_type == 0 && edge2_type == 0) motif_type_star = 0;
            else if (edge1_type == 0 && edge2_type == 1) motif_type_star = 1;
            else if (edge1_type == 1 && edge2_type == 0) motif_type_star = 1;
            else if (edge1_type == 0 && edge2_type == 2) motif_type_star = 2;
            else if (edge1_type == 2 && edge2_type == 0) motif_type_star = 2;
            else if (edge1_type == 0 && edge2_type == 3) motif_type_star = 3;
            else if (edge1_type == 3 && edge2_type == 0) motif_type_star = 3;
            else if (edge1_type == 0 && edge2_type == 4) motif_type_star = 4;
            else if (edge1_type == 4 && edge2_type == 0) motif_type_star = 4;
            else if (edge1_type == 1 && edge2_type == 1) motif_type_star = 5;
            else if (edge1_type == 1 && edge2_type == 2) motif_type_star = 6;
            else if (edge1_type == 2 && edge2_type == 1) motif_type_star = 6;
            else if (edge1_type == 1 && edge2_type == 3) motif_type_star = 7;
            else if (edge1_type == 3 && edge2_type == 1) motif_type_star = 7;
            else if (edge1_type == 1 && edge2_type == 4) motif_type_star = 8;
            else if (edge1_type == 4 && edge2_type == 1) motif_type_star = 8;
            else if (edge1_type == 2 && edge2_type == 2) motif_type_star = 9;
            else if (edge1_type == 2 && edge2_type == 3) motif_type_star = 10;
            else if (edge1_type == 3 && edge2_type == 2) motif_type_star = 10;
            else if (edge1_type == 2 && edge2_type == 4) motif_type_star = 11;
            else if (edge1_type == 4 && edge2_type == 2) motif_type_star = 11;
            else if (edge1_type == 3 && edge2_type == 3) motif_type_star = 12;
            else if (edge1_type == 3 && edge2_type == 4) motif_type_star = 13;
            else if (edge1_type == 4 && edge2_type == 3) motif_type_star = 13;
            else if (edge1_type == 4 && edge2_type == 4) motif_type_star = 14;

            if (motif_type_star != -1) {
                atomicAdd(&counts[motif_type_star], (unsigned int) 1);
            }
        }

        // Chain Motif Mapping
        int start_b = adj_offset[b];
        int end_b = adj_offset[b + 1];
        if (start_b >= end_b) continue;

        for (int j = start_b; j < end_b; j++) {
            int c = adj_list[j];
            int edge2_type = edge_types[j];
            if (c < 0 || c >= num_nodes) continue;

            int motif_type_chain = -1;
            if (edge1_type == 0 && edge2_type == 0) motif_type_chain = 15;
            else if (edge1_type == 1 && edge2_type == 0) motif_type_chain = 16;
            else if (edge1_type == 1 && edge2_type == 1) motif_type_chain = 17;
            else if (edge1_type == 1 && edge2_type == 2) motif_type_chain = 18;
            else if (edge1_type == 2 && edge2_type == 1) motif_type_chain = 19;
            else if (edge1_type == 2 && edge2_type == 0) motif_type_chain = 20;
            else if (edge1_type == 2 && edge2_type == 2) motif_type_chain = 21;
            else if (edge1_type == 1 && edge2_type == 3) motif_type_chain = 22;
            else if (edge1_type == 3 && edge2_type == 1) motif_type_chain = 23;
            else if (edge1_type == 2 && edge2_type == 3) motif_type_chain = 24;
            else if (edge1_type == 3 && edge2_type == 2) motif_type_chain = 25;
            else if (edge1_type == 3 && edge2_type == 0) motif_type_chain = 26;
            else if (edge1_type == 3 && edge2_type == 3) motif_type_chain = 27;
            else if (edge1_type == 1 && edge2_type == 4) motif_type_chain = 28;
            else if (edge1_type == 4 && edge2_type == 1) motif_type_chain = 29;
            else if (edge1_type == 2 && edge2_type == 4) motif_type_chain = 30;
            else if (edge1_type == 4 && edge2_type == 2) motif_type_chain = 31;
            else if (edge1_type == 3 && edge2_type == 4) motif_type_chain = 32;
            else if (edge1_type == 4 && edge2_type == 3) motif_type_chain = 33;
            else if (edge1_type == 4 && edge2_type == 0) motif_type_chain = 34;
            else if (edge1_type == 4 && edge2_type == 4) motif_type_chain = 35;
            
            
            
            if (motif_type_chain != -1) {
                atomicAdd(&counts[motif_type_chain], (unsigned int) 1);
            }
        }
    }
}

// Main function
int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input_graph_csv> <output_motif_counts> <num_swaps> <output_zscores>\n";
        return 1;
    }

    std::string input_graph = argv[1];
    std::string output_counts = argv[2];
    int num_swaps = std::stoi(argv[3]);
    std::string output_zscores = argv[4];

    std::cout << "Starting motif classification with null model generation...\n";
    
    int num_nodes = 0;
    std::vector<Edge> edges = load_graph_from_csv(input_graph, num_nodes);
    
    // Compute real motif counts
    std::vector<int> adj_list, adj_offset, edge_types;
    build_adjacency_list(edges, adj_list, adj_offset, edge_types, num_nodes);

    thrust::device_vector<int> d_adj_list = adj_list;
    thrust::device_vector<int> d_adj_offset = adj_offset;
    thrust::device_vector<int> d_edge_types = edge_types;
    thrust::device_vector<unsigned int> d_real_counts(36, 0);

    std::cout << "Computing real motif counts...\n";
    classify_subtrees<<<(num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_adj_offset.data()), 
        thrust::raw_pointer_cast(d_adj_list.data()), 
        thrust::raw_pointer_cast(d_edge_types.data()), 
        num_nodes, 
        thrust::raw_pointer_cast(d_real_counts.data())
    );
    cudaDeviceSynchronize();
    
    thrust::host_vector<unsigned int> real_counts = d_real_counts;
    printf("Real motif counts: \n");
    for (int i = 0; i < 36; i++) {
        printf(" %u", real_counts[i]);
    }
    printf("\n");
    save_motif_counts(output_counts, real_counts);
    
    std::vector<std::vector<unsigned int>> null_motif_counts(NUM_NULL_MODELS, std::vector<unsigned int>(36, 0));
    
    for (int i = 0; i < NUM_NULL_MODELS; i++) {
        std::vector<Edge> null_edges = generate_null_model(edges, num_swaps, num_nodes);
        build_adjacency_list(null_edges, adj_list, adj_offset, edge_types, num_nodes);
        
        d_adj_list = adj_list;
        d_adj_offset = adj_offset;
        d_edge_types = edge_types;
        thrust::device_vector<unsigned int> d_counts(36, 0);
        
        std::cout << "Running null model iteration " << i + 1 << "...\n";
        classify_subtrees<<<(num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(d_adj_offset.data()), 
            thrust::raw_pointer_cast(d_adj_list.data()), 
            thrust::raw_pointer_cast(d_edge_types.data()), 
            num_nodes, 
            thrust::raw_pointer_cast(d_counts.data())
        );
        cudaDeviceSynchronize();
        
        thrust::host_vector<int> h_counts = d_counts;
        for (int j = 0; j < 36; j++) {
            null_motif_counts[i][j] = h_counts[j];
            //printf("Null model %d, Motif %d: %d\n", i, j, h_counts[j]);
        }
    }
    
    std::cout << "Computing Z-scores...\n";
    compute_z_scores(real_counts, null_motif_counts, output_zscores);
    std::cout << "Z-score results saved to " << output_zscores << "\n";
    
    return 0;
}