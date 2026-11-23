#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
template<std::size_t width>
using Mask = std::array<uint64_t, width>;

template<std::size_t width>
void insert(Mask<width>& mask, int v) { mask[v >> 6] |= uint64_t(1) << (v & 63); }
template<std::size_t width>
int count(const Mask<width>& mask) {
    int k = 0;
    for (auto word : mask) k += __builtin_popcountll(word);
    return k;
}
template<std::size_t width>
Mask<width> both(const Mask<width>& a, const Mask<width>& b) {
    Mask<width> x;
    for (int w = 0; w < width; ++w) x[w] = a[w] & b[w];
    return x;
}
template<std::size_t width>
Mask<width> without(const Mask<width>& a, const Mask<width>& b) {
    Mask<width> x;
    for (int w = 0; w < width; ++w) x[w] = a[w] & ~b[w];
    return x;
}

struct Edge {
    int u, v;
    float affinity;
};

// Exact clique expansion on a residual compatibility graph of at most 64 nodes.
// Greedy coloring bounds how many compatible vertices can still be added.
class ResidualSearch {
    const std::array<uint64_t,64>& compatible;
    int visits = 0;
    Clock::time_point deadline;
    bool stop = false;
public:
    uint64_t optimum;
    int size;
    ResidualSearch(const std::array<uint64_t,64>& adjacency, uint64_t incumbent, Clock::time_point end)
        : compatible(adjacency), deadline(end), optimum(incumbent),
          size(__builtin_popcountll(incumbent) - 1) {}

    void expand(uint64_t candidates, uint64_t chosen, int depth) {
        if (++visits >= 2500 || ((visits & 31) == 1 && Clock::now() >= deadline)) {
            stop = true;
            return;
        }
        if (depth + __builtin_popcountll(candidates) <= size) return;
        unsigned char order[64], bound[64];
        int length = 0, colors = 0;
        for (uint64_t rest = candidates; rest;) {
            uint64_t possible = rest;
            ++colors;
            while (possible) {
                int v = __builtin_ctzll(possible);
                uint64_t bit = uint64_t(1) << v;
                order[length] = v;
                bound[length++] = colors;
                rest ^= bit;
                possible &= ~compatible[v] & ~bit;
            }
        }
        for (int i=length-1; i>=0 && !stop; --i) {
            if (depth + bound[i] <= size) return;
            int v = order[i];
            uint64_t bit = uint64_t(1) << v, selection = chosen | bit;
            if (depth + 1 > size) { size = depth + 1; optimum = selection; }
            uint64_t next = candidates & compatible[v];
            if (next) expand(next, selection, depth+1);
            candidates ^= bit;
        }
    }
};

template<int width>
class CliqueSearch {
    using Mask = std::array<uint64_t, width>;
    static constexpr int limit = width * 64;
    int n;
    std::vector<Mask> closed, groups;
    Mask all{};
    const float* prior;
    std::mt19937 random;
    Clock::time_point deadline;
    int visits = 0, quota = 0;
    bool interrupted = false;
    std::vector<int> path;

    // Every merge checks the complete bipartite cross edges. The resulting
    // disjoint cliques bound the number of additional independent vertices.
    void partition(const int32_t* row, const int32_t* col) {
        std::vector<int> owner(n), degree(n);
        std::iota(owner.begin(), owner.end(), 0);
        std::vector<Mask> members(n), common = closed;
        std::vector<Edge> edges;
        edges.reserve(row[n] / 2);
        for (int v = 0; v < n; ++v) {
            degree[v] = count(closed[v]);
            insert(members[v], v);
        }
        for (int u = 0; u < n; ++u) {
            for (int i = row[u]; i < row[u+1]; ++i) {
                int v = col[i];
                if (v <= u) continue;
                int shared = count(both(closed[u], closed[v]));
                float affinity = float(shared) / (degree[u] + degree[v] - shared);
                edges.push_back({u, v, affinity});
            }
        }
        std::shuffle(edges.begin(), edges.end(), random);
        std::stable_sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            return a.affinity > b.affinity;
        });
        for (const Edge& edge : edges) {
            int a = owner[edge.u], b = owner[edge.v];
            if (a == b || count(without(members[b], common[a]))) continue;
            for (int w = 0; w < width; ++w) {
                uint64_t bits = members[b][w];
                while (bits) {
                    owner[64*w + __builtin_ctzll(bits)] = a;
                    bits &= bits - 1;
                }
                members[a][w] |= members[b][w];
                members[b][w] = 0;
                common[a][w] &= common[b][w];
            }
        }
        for (const Mask& mask : members) if (count(mask)) groups.push_back(mask);
    }

    void branch(const Mask& free) {
        ++visits;
        if (visits >= quota || ((visits & 7) == 1 && Clock::now() >= deadline)) {
            interrupted = true;
            return;
        }
        if (path.size() > best.size()) best = path;
        int needed = int(best.size() + 1 - path.size());
        int live = 0, smallest = n+1, choice = -1;
        for (int j = 0; j < int(groups.size()); ++j) {
            int size = count(both(free, groups[j]));
            if (size == 0) continue;
            ++live;
            if (size < smallest) {
                smallest = size;
                choice = j;
            }
        }
        if (live < needed) return;
        Mask domain = both(free, groups[choice]);
        std::array<std::pair<float,int>, limit> ranked;
        int length = 0;
        for (int w = 0; w < width; ++w) {
            uint64_t bits = domain[w];
            while (bits) {
                int v = 64*w + __builtin_ctzll(bits);
                float cost = count(both(free, closed[v])) - prior[v] + float(random() % 1000) * .001f;
                ranked[length++] = {cost, v};
                bits &= bits-1;
            }
        }
        std::sort(ranked.begin(), ranked.begin()+length);
        for (int i = 0; i < length && !interrupted; ++i) {
            int v = ranked[i].second;
            path.push_back(v);
            branch(without(free, closed[v]));
            path.pop_back();
            if (int(path.size()) + live <= int(best.size())) return;
        }
        if (!interrupted && int(path.size()) + live > int(best.size()) + 1)
            branch(without(free, domain));
    }

    void repair_neighborhoods() {
        std::vector<int> current = best;
        for (int round=0; Clock::now() < deadline; ++round) {
            if (round % 11 == 0) current = best;
            std::shuffle(current.begin(), current.end(), random);
            Mask free = all;
            int fixed = 0;
            int cap = 42 + int(random() % 23);
            while (count(free) > cap && fixed < int(current.size())) {
                free = without(free, closed[current[fixed++]]);
            }
            if (count(free) > 64 || fixed == int(current.size())) continue;
            std::vector<int> vertices;
            for (int w=0; w<width; ++w) {
                uint64_t bits = free[w];
                while (bits) {
                    vertices.push_back(64*w + __builtin_ctzll(bits));
                    bits &= bits-1;
                }
            }
            std::shuffle(vertices.begin(),vertices.end(),random);
            std::array<uint64_t,64> adj{};
            uint64_t start = 0;
            Mask original{};
            for (int j=fixed; j<int(current.size()); ++j) insert(original,current[j]);
            for (int i=0; i<int(vertices.size()); ++i) {
                int v=vertices[i];
                if ((original[v>>6] >> (v&63)) & 1) start |= uint64_t(1) << i;
                for (int j=0; j<i; ++j) {
                    int u=vertices[j];
                    if (((closed[v][u>>6] >> (u&63)) & 1) == 0) {
                        adj[i] |= uint64_t(1) << j;
                        adj[j] |= uint64_t(1) << i;
                    }
                }
            }
            ResidualSearch repair(adj,start,deadline);
            uint64_t candidates = vertices.size() == 64 ? ~uint64_t(0) : (uint64_t(1)<<vertices.size())-1;
            repair.expand(candidates,0,0);
            if (repair.size < __builtin_popcountll(start)) continue;
            current.resize(fixed);
            for (uint64_t bits = repair.optimum; bits; bits &= bits-1)
                current.push_back(vertices[__builtin_ctzll(bits)]);
            if (current.size() > best.size()) best = current;
        }
    }
public:
    std::vector<int> best;
    int upper;
    CliqueSearch(int nodes, const int32_t* row, const int32_t* col, const float* score,
                 unsigned seed, Clock::time_point stop, const int32_t* incumbent, int size)
        : n(nodes), closed(n), prior(score), random(seed), deadline(stop),
          best(incumbent, incumbent + size) {
        for (int u=0; u<n; ++u) {
            insert(all, u);
            insert(closed[u], u);
            for (int i=row[u]; i<row[u+1]; ++i) insert(closed[u],col[i]);
        }
        partition(row, col);
        upper = int(groups.size());
    }
    void solve() {
        if (n > 320 && upper > 2 * std::max(1, int(best.size()))) {
            repair_neighborhoods();
            return;
        }
        for (int restart=0; Clock::now()<deadline && best.size()<groups.size(); ++restart) {
            std::shuffle(groups.begin(), groups.end(), random);
            std::vector<int> anchor = best;
            std::shuffle(anchor.begin(), anchor.end(), random);
            path.clear();
            Mask free = all;
            int fixed = restart % 4 == 0 ? 0 : int(anchor.size()) * (2 + restart % 5) / 10;
            for (int j=0; j<fixed; ++j) {
                path.push_back(anchor[j]);
                free = without(free, closed[anchor[j]]);
            }
            visits = 0;
            quota = fixed ? 1500 : 6000;
            interrupted = false;
            branch(free);
            if (!fixed && !interrupted) {
                upper = int(best.size());
                break;
            }
        }
    }
};
template<int width>
int refine(int n, const int32_t* row, const int32_t* col, const float* priority,
           uint32_t seed, Clock::time_point end, int32_t* nodes, int size, int32_t* upper) {
    CliqueSearch<width> search(n, row, col, priority, seed, end, nodes, size);
    search.solve();
    std::copy(search.best.begin(), search.best.end(), nodes);
    *upper = search.upper;
    return int(search.best.size());
}
}

extern "C" int clique_refine(int n, const int32_t* crow, const int32_t* col,
                            const float* priority, uint32_t seed, double seconds,
                            int32_t* nodes, int current_size, int32_t* upper) {
    *upper = n;
    if (n > 1216 || n == 0 || seconds <= 0.0) return current_size;
    auto end = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    if (n <= 320) return refine<5>(n, crow, col, priority, seed, end, nodes, current_size, upper);
    return refine<19>(n, crow, col, priority, seed, end, nodes, current_size, upper);
}
