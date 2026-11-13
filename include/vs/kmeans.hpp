// Lloyd's k-means, used for IVF's coarse lists and PQ's codebooks.
#pragma once

#include <random>

#include "common.hpp"

namespace vs {

struct KMeansResult {
    Matrix centroids;
    std::vector<double> inertia;  // sum of squared distances after each iteration
};

// Trains on at most `sample` rows of x (all if 0). Starts from k distinct random rows; an
// empty cluster is restarted at a random row. Deterministic for a given seed.
inline KMeansResult kmeans(const Matrix& x, std::size_t k, int iters, std::uint64_t seed, std::size_t sample = 0) {
    std::mt19937_64 rng(seed);
    std::vector<std::size_t> rows(x.n);
    for (std::size_t i = 0; i < x.n; ++i) rows[i] = i;
    if (sample && sample < x.n) {
        std::shuffle(rows.begin(), rows.end(), rng);
        rows.resize(sample);
    }
    if (rows.size() < k) throw std::invalid_argument("kmeans: fewer points than clusters");
    std::vector<std::size_t> pick = rows;
    std::shuffle(pick.begin(), pick.end(), rng);
    KMeansResult r{Matrix(k, x.d), {}};
    for (std::size_t j = 0; j < k; ++j) std::copy_n(x.row(pick[j]), x.d, r.centroids.row(j));

    std::vector<std::size_t> assign(rows.size());
    std::vector<double> sums(k * x.d);
    std::vector<std::size_t> counts(k);
    std::uniform_int_distribution<std::size_t> any(0, rows.size() - 1);
    for (int it = 0; it < iters; ++it) {
        double inertia = 0;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const float* p = x.row(rows[i]);
            assign[i] = nearest(r.centroids, p);
            inertia += l2sq(r.centroids.row(assign[i]), p, x.d);
        }
        r.inertia.push_back(inertia);
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const float* p = x.row(rows[i]);
            double* s = sums.data() + assign[i] * x.d;
            for (std::size_t t = 0; t < x.d; ++t) s[t] += p[t];
            ++counts[assign[i]];
        }
        for (std::size_t j = 0; j < k; ++j) {
            float* c = r.centroids.row(j);
            if (counts[j] == 0) {
                std::copy_n(x.row(rows[any(rng)]), x.d, c);
                continue;
            }
            for (std::size_t t = 0; t < x.d; ++t)
                c[t] = static_cast<float>(sums[j * x.d + t] / static_cast<double>(counts[j]));
        }
    }
    return r;
}

}  // namespace vs
