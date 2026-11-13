// Correctness on small random data; every index against exact search.
#include <random>
#include <set>

#include "check.hpp"
#include "vs/hnsw.hpp"
#include "vs/ivf.hpp"

using namespace vs;

// Points around a few centres, normalised to unit length like the benchmark data.
static Matrix clustered(std::size_t n, std::size_t d, std::uint64_t seed, std::size_t centres = 20) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    Matrix c(centres, d), x(n, d);
    for (float& v : c.v) v = g(rng);
    for (std::size_t i = 0; i < n; ++i) {
        const float* cc = c.row(i % centres);
        float norm = 0;
        for (std::size_t t = 0; t < d; ++t) {
            x.row(i)[t] = cc[t] + 0.4f * g(rng);
            norm += x.row(i)[t] * x.row(i)[t];
        }
        for (std::size_t t = 0; t < d; ++t) x.row(i)[t] /= std::sqrt(norm);
    }
    return x;
}

static double recall(const std::vector<Hit>& got, const std::vector<Hit>& truth) {
    std::set<id_t> t;
    for (const Hit& h : truth) t.insert(h.id);
    std::size_t n = 0;
    for (const Hit& h : got) n += t.count(h.id);
    return static_cast<double>(n) / static_cast<double>(truth.size());
}

TEST(topk_keeps_the_k_smallest) {
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<float> u(0, 1);
    std::vector<Hit> all;
    TopK top(10);
    for (id_t i = 0; i < 1000; ++i) {
        float d = u(rng);
        all.push_back({d, i});
        top.push(d, i);
    }
    std::sort(all.begin(), all.end());
    all.resize(10);
    CHECK(top.sorted() == all);
}

TEST(kmeans_inertia_never_rises) {
    Matrix x = clustered(3000, 16, 2);
    KMeansResult r = kmeans(x, 32, 15, 7);
    for (std::size_t i = 1; i < r.inertia.size(); ++i) CHECK(r.inertia[i] <= r.inertia[i - 1] * (1 + 1e-9));
    CHECK(r.inertia.back() < 0.7 * r.inertia.front());
}

TEST(ivf_probing_every_list_is_exact) {
    Matrix x = clustered(3000, 16, 3), q = clustered(50, 16, 4);
    IVFFlat ivf(x, 32, 5);
    for (std::size_t i = 0; i < q.n; ++i) CHECK(ivf.search(q.row(i), 10, 32) == exact_search(x, q.row(i), 10));
}

TEST(ivf_recall_rises_with_nprobe) {
    Matrix x = clustered(4000, 16, 5), q = clustered(100, 16, 6);
    IVFFlat ivf(x, 64, 5);
    double prev = 0;
    for (std::size_t p : std::initializer_list<std::size_t>{1, 4, 16, 64}) {
        double r = 0;
        for (std::size_t i = 0; i < q.n; ++i) r += recall(ivf.search(q.row(i), 10, p), exact_search(x, q.row(i), 10));
        r /= static_cast<double>(q.n);
        CHECK(r >= prev - 1e-9);
        prev = r;
    }
    CHECK(prev == 1.0);
}

TEST(pq_distance_is_the_distance_to_the_decoded_vector) {
    Matrix x = clustered(3000, 16, 7), q = clustered(20, 16, 8);
    PQ pq(16, 4);
    pq.train(x, 9, 10);
    std::vector<std::uint8_t> code(4);
    std::vector<float> dec(16), t(4 * PQ::ks);
    for (std::size_t i = 0; i < q.n; ++i) {
        pq.table(q.row(i), t.data());
        for (std::size_t j = 0; j < 50; ++j) {
            pq.encode(x.row(j), code.data());
            pq.decode(code.data(), dec.data());
            CHECK(std::abs(PQ::adc(t.data(), code.data(), 4) - l2sq(dec.data(), q.row(i), 16)) < 1e-4f);
        }
    }
}

TEST(pq_code_is_the_nearest_centroid_per_part) {
    // encoding then decoding must be at least as close as any other code in each sub-space
    Matrix x = clustered(2000, 8, 10);
    PQ pq(8, 2);
    pq.train(x, 11, 10);
    std::vector<std::uint8_t> code(2), other(2);
    std::vector<float> a(8), b(8);
    for (std::size_t i = 0; i < 20; ++i) {
        pq.encode(x.row(i), code.data());
        pq.decode(code.data(), a.data());
        for (std::size_t c = 0; c < PQ::ks; c += 17) {
            other = code;
            other[0] = static_cast<std::uint8_t>(c);
            pq.decode(other.data(), b.data());
            CHECK(l2sq(a.data(), x.row(i), 8) <= l2sq(b.data(), x.row(i), 8) + 1e-6f);
        }
    }
}

TEST(ivfpq_with_full_probe_and_rerank_is_exact) {
    Matrix x = clustered(3000, 16, 12), q = clustered(30, 16, 13);
    IVFPQ ix(x, 16, 4, 14);
    for (std::size_t i = 0; i < q.n; ++i) CHECK(ix.search(q.row(i), 10, 16, x.n) == exact_search(x, q.row(i), 10));
}

TEST(ivfpq_compression_costs_some_recall_rerank_wins_it_back) {
    Matrix x = clustered(5000, 16, 15), q = clustered(100, 16, 16);
    IVFPQ ix(x, 32, 4, 17);
    double plain = 0, reranked = 0;
    for (std::size_t i = 0; i < q.n; ++i) {
        auto truth = exact_search(x, q.row(i), 10);
        plain += recall(ix.search(q.row(i), 10, 32), truth);
        reranked += recall(ix.search(q.row(i), 10, 32, 200), truth);
    }
    CHECK(plain / 100 < 0.99);
    CHECK(reranked / 100 > plain / 100);
    CHECK(reranked / 100 > 0.97);
}

TEST(hnsw_finds_the_neighbours) {
    Matrix x = clustered(5000, 16, 18), q = clustered(200, 16, 19);
    HNSW h(x, 12, 100, 20);
    double r = 0;
    for (std::size_t i = 0; i < q.n; ++i) r += recall(h.search(q.row(i), 10, 200), exact_search(x, q.row(i), 10));
    CHECK(r / 200 >= 0.98);
}

TEST(hnsw_recall_rises_with_ef) {
    Matrix x = clustered(5000, 16, 21), q = clustered(200, 16, 22);
    HNSW h(x, 8, 50, 23);
    double prev = 0;
    for (std::size_t ef : std::initializer_list<std::size_t>{10, 40, 160}) {
        double r = 0;
        for (std::size_t i = 0; i < q.n; ++i) r += recall(h.search(q.row(i), 10, ef), exact_search(x, q.row(i), 10));
        CHECK(r >= prev);
        prev = r;
    }
}

TEST(hnsw_graph_invariants) {
    Matrix x = clustered(3000, 16, 24);
    HNSW h(x, 8, 50, 25);
    std::size_t at_top = 0;
    for (id_t i = 0; i < x.n; ++i) {
        CHECK(h.degree(i, 0) >= 1 && h.degree(i, 0) <= 16);  // 2M on layer 0
        for (int lc = 1; lc <= h.level(i); ++lc) CHECK(h.degree(i, lc) <= 8);
        for (id_t n : h.neighbours(i, 0)) CHECK(n != i && n < x.n);
        at_top += h.level(i) == h.top_level();
    }
    CHECK(at_top >= 1);
    // every node is reachable on layer 0 from node 0
    std::vector<bool> seen(x.n);
    std::vector<id_t> stack{0};
    seen[0] = true;
    std::size_t reached = 1;
    while (!stack.empty()) {
        id_t c = stack.back();
        stack.pop_back();
        for (id_t n : h.neighbours(c, 0))
            if (!seen[n]) {
                seen[n] = true;
                ++reached;
                stack.push_back(n);
            }
    }
    CHECK(reached == x.n);
}

TEST(hnsw_is_deterministic) {
    Matrix x = clustered(2000, 16, 26), q = clustered(20, 16, 27);
    HNSW a(x, 8, 50, 28), b(x, 8, 50, 28);
    for (std::size_t i = 0; i < q.n; ++i) CHECK(a.search(q.row(i), 10, 50) == b.search(q.row(i), 10, 50));
}

int main() { return check::run_all(); }
