// HNSW: hierarchical navigable small world graphs (Malkov and Yashunin, 2018).
//
// Every vector is a node. Each node gets a random top layer (most only layer 0, a few reach
// higher, geometrically). On every layer it belongs to, a node links to up to M neighbours
// (2M on layer 0). A search starts at the single top node, walks greedily down the sparse
// upper layers to get close, then runs a best-first search with a candidate list of size ef
// on layer 0. Neighbours are chosen with the paper's heuristic (Algorithm 4): a candidate is
// kept only if it is closer to the new node than to every neighbour kept so far, which keeps
// links pointing in different directions rather than all into one cluster.
#pragma once

#include <queue>
#include <random>

#include "common.hpp"

namespace vs {

class HNSW {
public:
    HNSW(const Matrix& base, std::size_t M, std::size_t ef_construction, std::uint64_t seed)
        : x_(base), M_(M), M0_(2 * M), efc_(ef_construction), mL_(1.0 / std::log(static_cast<double>(M))),
          level_(base.n, 0), links0_(base.n * (2 * M)), count0_(base.n, 0), upper_(base.n), visited_(base.n, 0) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (std::size_t i = 0; i < base.n; ++i) {
            double r = 1.0 - u(rng);  // in (0, 1]
            level_[i] = static_cast<int>(std::floor(-std::log(r) * mL_));
        }
        for (std::size_t i = 0; i < base.n; ++i) insert(static_cast<id_t>(i));
    }

    std::vector<Hit> search(const float* q, std::size_t k, std::size_t ef) const {
        if (x_.n == 0) return {};
        id_t ep = entry_;
        for (int lc = top_; lc > 0; --lc) ep = greedy(q, ep, lc);
        std::vector<Hit> w = search_layer(q, {ep}, std::max(ef, k), 0);
        if (w.size() > k) w.resize(k);
        return w;
    }

    int top_level() const { return top_; }
    int level(id_t i) const { return level_[i]; }
    std::size_t degree(id_t i, int lc) const { return neighbours(i, lc).size(); }
    std::vector<id_t> neighbours(id_t i, int lc) const {
        if (lc == 0) return {links0_.begin() + static_cast<std::ptrdiff_t>(i * M0_),
                             links0_.begin() + static_cast<std::ptrdiff_t>(i * M0_ + count0_[i])};
        return upper_[i][static_cast<std::size_t>(lc - 1)];
    }
    std::size_t bytes() const {
        std::size_t b = links0_.size() * sizeof(id_t) + count0_.size() * sizeof(std::uint32_t);
        for (const auto& u : upper_)
            for (const auto& l : u) b += l.size() * sizeof(id_t);
        return b;
    }

private:
    float dist(id_t a, const float* q) const { return l2sq(x_.row(a), q, x_.d); }

    void set_links(id_t i, int lc, const std::vector<id_t>& ns) {
        if (lc == 0) {
            std::copy(ns.begin(), ns.end(), links0_.begin() + static_cast<std::ptrdiff_t>(i * M0_));
            count0_[i] = static_cast<std::uint32_t>(ns.size());
        } else {
            upper_[i][static_cast<std::size_t>(lc - 1)] = ns;
        }
    }

    // One step at a time towards q on an upper layer, until no neighbour is closer.
    id_t greedy(const float* q, id_t ep, int lc) const {
        float d = dist(ep, q);
        for (bool moved = true; moved;) {
            moved = false;
            for (id_t n : neighbours(ep, lc)) {
                float dn = dist(n, q);
                if (dn < d) {
                    d = dn;
                    ep = n;
                    moved = true;
                }
            }
        }
        return ep;
    }

    // Best-first search keeping the ef closest nodes found (Algorithm 2). Returns them nearest first.
    std::vector<Hit> search_layer(const float* q, const std::vector<id_t>& eps, std::size_t ef, int lc) const {
        if (++epoch_ == 0) {  // wrapped: reset the marks
            std::fill(visited_.begin(), visited_.end(), 0);
            epoch_ = 1;
        }
        auto far_first = [](const Hit& a, const Hit& b) { return a < b; };
        auto near_first = [](const Hit& a, const Hit& b) { return b < a; };
        std::priority_queue<Hit, std::vector<Hit>, decltype(near_first)> cand(near_first);
        std::priority_queue<Hit, std::vector<Hit>, decltype(far_first)> found(far_first);
        for (id_t e : eps) {
            visited_[e] = epoch_;
            Hit h{dist(e, q), e};
            cand.push(h);
            found.push(h);
        }
        while (!cand.empty()) {
            Hit c = cand.top();
            cand.pop();
            if (c.dist > found.top().dist) break;  // nothing left can improve the list
            for (id_t n : neighbours(c.id, lc)) {
                if (visited_[n] == epoch_) continue;
                visited_[n] = epoch_;
                float dn = dist(n, q);
                if (found.size() < ef || dn < found.top().dist) {
                    cand.push({dn, n});
                    found.push({dn, n});
                    if (found.size() > ef) found.pop();
                }
            }
        }
        std::vector<Hit> out;
        out.reserve(found.size());
        for (; !found.empty(); found.pop()) out.push_back(found.top());
        std::reverse(out.begin(), out.end());
        return out;
    }

    // Algorithm 4: candidates nearest first; keep one if it is closer to the base point than
    // to any neighbour already kept.
    std::vector<id_t> select(const std::vector<Hit>& candidates, std::size_t m) const {
        std::vector<id_t> kept;
        for (const Hit& c : candidates) {
            if (kept.size() >= m) break;
            bool diverse = true;
            for (id_t k : kept) {
                if (l2sq(x_.row(c.id), x_.row(k), x_.d) < c.dist) {
                    diverse = false;
                    break;
                }
            }
            if (diverse) kept.push_back(c.id);
        }
        return kept;
    }

    void insert(id_t i) {
        int l = level_[i];
        upper_[i].resize(static_cast<std::size_t>(l));
        if (i == 0) {
            entry_ = 0;
            top_ = l;
            return;
        }
        const float* q = x_.row(i);
        id_t ep = entry_;
        for (int lc = top_; lc > l; --lc) ep = greedy(q, ep, lc);
        std::vector<id_t> eps{ep};
        for (int lc = std::min(top_, l); lc >= 0; --lc) {
            std::vector<Hit> w = search_layer(q, eps, efc_, lc);
            std::size_t mmax = lc == 0 ? M0_ : M_;
            std::vector<id_t> ns = select(w, M_);
            set_links(i, lc, ns);
            for (id_t n : ns) {
                std::vector<id_t> nn = neighbours(n, lc);
                nn.push_back(i);
                if (nn.size() > mmax) {  // too many links: keep a diverse subset
                    std::vector<Hit> cs;
                    cs.reserve(nn.size());
                    for (id_t c : nn) cs.push_back({l2sq(x_.row(c), x_.row(n), x_.d), c});
                    std::sort(cs.begin(), cs.end());
                    nn = select(cs, mmax);
                }
                set_links(n, lc, nn);
            }
            eps.clear();
            for (const Hit& h : w) eps.push_back(h.id);
        }
        if (l > top_) {
            top_ = l;
            entry_ = i;
        }
    }

    const Matrix& x_;
    std::size_t M_, M0_, efc_;
    double mL_;
    std::vector<int> level_;
    std::vector<id_t> links0_;  // layer 0: a fixed block of 2M slots per node
    std::vector<std::uint32_t> count0_;
    std::vector<std::vector<std::vector<id_t>>> upper_;  // layers 1+: only for the few nodes that reach them
    id_t entry_ = 0;
    int top_ = -1;
    mutable std::vector<std::uint32_t> visited_;  // search is single-threaded
    mutable std::uint32_t epoch_ = 0;
};

}  // namespace vs
