#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Publish = void (*)(int, const int*);

class Random {
    uint64_t state;
public:
    explicit Random(uint32_t seed) : state(seed + 1ull) {}
    uint32_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return (state * 2685821657736338717ull) >> 32;
    }
    int below(int size) { return (uint64_t(next()) * size) >> 32; }
    float unit() { return (next() >> 8) * (1.f / 16777216.f); }
};

class Pool {
    std::vector<int> positions;
public:
    std::vector<int> nodes;
    explicit Pool(int n) : positions(n) { nodes.reserve(n); }
    void insert(int v) {
        positions[v] = nodes.size();
        nodes.push_back(v);
    }
    void erase(int v) {
        int p = positions[v];
        positions[nodes.back()] = p;
        nodes[p] = nodes.back();
        nodes.pop_back();
    }
    int sample(Random& rng) const { return nodes[rng.below(nodes.size())]; }
};

class Search {
    int n, words;
    const int* row;
    const int* col;
    Random rng;
    Pool available, one, selected;
    std::vector<unsigned char> in;
    std::vector<int> conflicts, owner, tabu, frequency;
    std::vector<uint64_t> adjacent;
    std::vector<float> priority;
    std::vector<int> best;
    Publish publish;
    int step = 0;

    void insert(int v) {
        available.erase(v);
        in[v] = 1;
        selected.insert(v);
        for (int e = row[v]; e < row[v+1]; ++e) {
            int u = col[e], before = conflicts[u]++;
            owner[u] ^= v;
            if (before == 0) {
                available.erase(u);
                one.insert(u);
            } else if (before == 1) one.erase(u);
        }
    }
    void remove(int v) {
        in[v] = 0;
        selected.erase(v);
        available.insert(v);
        for (int e = row[v]; e < row[v+1]; ++e) {
            int u = col[e], before = conflicts[u]--;
            owner[u] ^= v;
            if (before == 1) {
                one.erase(u);
                available.insert(u);
            } else if (before == 2) one.insert(u);
        }
    }
    int pick(const Pool& pool, int samples, bool use_tabu) {
        int choice = -1;
        float high = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < samples; ++i) {
            int v = pool.sample(rng);
            if (use_tabu && tabu[v] > step) continue;
            float score = rng.unit() + priority[v] - .09f * frequency[v];
            if (score > high) { high = score; choice = v; }
        }
        return choice;
    }
    void fill() {
        while (!available.nodes.empty()) insert(pick(available, 5, false));
    }
    bool connected(int v, int u) const {
        return adjacent[size_t(v)*words + (u>>6)] & (1ull << (u&63));
    }
    void record() {
        if (selected.nodes.size() > best.size()) {
            best = selected.nodes;
            publish(best.size(), best.data());
        }
    }
    void force(int v) {
        for (int i = int(selected.nodes.size())-1; i >= 0; --i) {
            int u = selected.nodes[i];
            if (connected(v,u)) remove(u);
        }
        insert(v);
    }
    void restore_best() {
        while (!selected.nodes.empty()) remove(selected.nodes.back());
        for (int v : best) insert(v);
    }
    void kick(int strength) {
        for (int k = 0; k < strength; ++k) {
            int choice = -1, low = n+1;
            for (int i = 0; i < 12; ++i) {
                int v = rng.below(n);
                if (in[v] || conflicts[v] < 2) continue;
                if (conflicts[v] < low) { low = conflicts[v]; choice = v; }
            }
            if (choice >= 0) force(choice);
        }
        fill();
    }
    bool exchange() {
        if (one.nodes.empty()) return false;
        int v = pick(one, 5, true);
        if (v < 0) return false;
        int u = owner[v], second = -1;
        for (int e = row[u]; e < row[u+1]; ++e) {
            int w = col[e];
            if (w != v && conflicts[w] == 1 && !connected(w,v)) {
                second = w; break;
            }
        }
        remove(u);
        insert(v);
        tabu[u] = step + 3 + rng.below(8);
        if (second >= 0) { insert(second); fill(); }
        return true;
    }
public:
    Search(int count, const int* offsets, const int* neighbors, const float* rank,
           float weight, uint32_t seed, Publish emit)
      : n(count), words((n+63)/64), row(offsets), col(neighbors), rng(seed),
        available(n), one(n), selected(n), in(n), conflicts(n), owner(n), tabu(n),
        frequency(n), adjacent(size_t(n)*words), priority(n), publish(emit) {
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return rank[a]<rank[b];});
        for (int v = 0; v < n; ++v) {
            available.insert(v);
            for (int e=row[v];e<row[v+1];++e) adjacent[size_t(v)*words+(col[e]>>6)] |= 1ull << (col[e]&63);
        }
        for (int begin=0; begin<n;) {
            int end=begin+1;
            while (end<n && rank[order[begin]]==rank[order[end]]) ++end;
            for (int i=begin;i<end;++i) priority[order[i]]=weight*float(begin+end-1)/(2*n);
            begin=end;
        }
    }
    int solve(Clock::time_point end, int plateau, int perturb, const int* initial, int size) {
        if (size > 1) for (int i=0; i<size; ++i) insert(initial[i]);
        fill(); record();
        int stale=0, phase=0;
        while (Clock::now() < end && int(best.size()) < n) {
            ++step;
            if (stale >= plateau || !exchange()) {
                if ((++phase % 4)==0) restore_best();
                for (int v:selected.nodes) ++frequency[v];
                if (phase%12==0) for (int& f:frequency) f/=2;
                kick(1+rng.below(perturb));
                stale=0;
            }
            ++stale;
            if (selected.nodes.size()>best.size()) { record(); stale=0; }
        }
        return step;
    }
};
}

extern "C" int priority_seed(int n,const int* row,const int* col,uint32_t seed,int* nodes) {
    Random rng(seed);
    std::vector<int> degree(n), order(n);
    std::vector<unsigned char> active(n,1);
    std::iota(order.begin(),order.end(),0);
    for (int v=0;v<n;++v) {
        degree[v]=row[v+1]-row[v];
        std::swap(order[v],order[rng.below(v+1)]);
    }
    auto discard = [&](int v) {
        active[v]=0;
        for (int p=row[v];p<row[v+1];++p) --degree[col[p]];
    };
    int size=0;
    while (true) {
        int best=-1;
        for (int v:order) if (active[v] && (best<0 || degree[v]<degree[best])) best=v;
        if (best<0) return size;
        nodes[size++]=best;
        discard(best);
        for (int e=row[best];e<row[best+1];++e) if (active[col[e]]) discard(col[e]);
    }
}

extern "C" int priority_search(int n,const int* row,const int* col,const float* rank,
                               uint32_t seed,double seconds,float weight,int plateau,
                               int perturb,const int* initial,int size,Publish publish) {
    auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    if (n==0 || seconds<=0) return 0;
    Search search(n,row,col,rank,weight,seed,publish);
    return search.solve(end,plateau,perturb,initial,size);
}
