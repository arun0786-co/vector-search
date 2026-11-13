// IVF: k-means splits the vectors into lists; a query scans only the lists of its nearest
// centroids. IVF-PQ also compresses each vector to a few bytes with product quantization
// (Jegou, Douze and Schmid, 2011), coding the residual from its list's centroid.
#pragma once

#include <numeric>

#include "kmeans.hpp"

namespace vs {

// The `nprobe` centroids nearest to q, nearest first.
inline std::vector<std::size_t> probe(const Matrix& centroids, const float* q, std::size_t nprobe) {
    std::vector<Hit> d(centroids.n);
    for (std::size_t j = 0; j < centroids.n; ++j) d[j] = {l2sq(centroids.row(j), q, centroids.d), static_cast<id_t>(j)};
    nprobe = std::min(nprobe, d.size());
    std::partial_sort(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(nprobe), d.end());
    std::vector<std::size_t> out(nprobe);
    for (std::size_t i = 0; i < nprobe; ++i) out[i] = d[i].id;
    return out;
}

class IVFFlat {
public:
    IVFFlat(const Matrix& base, std::size_t nlist, std::uint64_t seed, std::size_t train = 200000, int iters = 20)
        : d_(base.d), centroids_(kmeans(base, nlist, iters, seed, train).centroids), ids_(nlist), vecs_(nlist) {
        for (std::size_t i = 0; i < base.n; ++i) {
            std::size_t l = nearest(centroids_, base.row(i));
            ids_[l].push_back(static_cast<id_t>(i));
            vecs_[l].insert(vecs_[l].end(), base.row(i), base.row(i) + d_);
        }
    }

    std::vector<Hit> search(const float* q, std::size_t k, std::size_t nprobe) const {
        TopK top(k);
        for (std::size_t l : probe(centroids_, q, nprobe)) {
            const float* v = vecs_[l].data();
            for (std::size_t i = 0; i < ids_[l].size(); ++i) top.push(l2sq(v + i * d_, q, d_), ids_[l][i]);
        }
        return top.sorted();
    }

    std::size_t bytes() const {
        std::size_t b = centroids_.v.size() * sizeof(float);
        for (std::size_t l = 0; l < ids_.size(); ++l) b += ids_[l].size() * (sizeof(id_t) + d_ * sizeof(float));
        return b;
    }

private:
    std::size_t d_;
    Matrix centroids_;
    std::vector<std::vector<id_t>> ids_;
    std::vector<std::vector<float>> vecs_;
};

// Product quantizer: d dimensions in m sub-vectors, each coded as one of 256 centroids.
class PQ {
public:
    static constexpr std::size_t ks = 256;

    PQ(std::size_t d, std::size_t m) : m_(m), ds_(d / m), book_(m * ks * (d / m)) {
        if (d % m != 0) throw std::invalid_argument("PQ: dimensions must divide into m parts");
    }

    void train(const Matrix& x, std::uint64_t seed, int iters) {
        for (std::size_t j = 0; j < m_; ++j) {
            Matrix sub(x.n, ds_);
            for (std::size_t i = 0; i < x.n; ++i) std::copy_n(x.row(i) + j * ds_, ds_, sub.row(i));
            Matrix c = kmeans(sub, ks, iters, seed + j).centroids;
            std::copy(c.v.begin(), c.v.end(), book_.begin() + static_cast<std::ptrdiff_t>(j * ks * ds_));
        }
    }

    void encode(const float* x, std::uint8_t* code) const {
        for (std::size_t j = 0; j < m_; ++j) {
            const float* s = x + j * ds_;
            std::size_t best = 0;
            float bd = std::numeric_limits<float>::infinity();
            for (std::size_t c = 0; c < ks; ++c) {
                float dist = l2sq(centroid(j, c), s, ds_);
                if (dist < bd) {
                    bd = dist;
                    best = c;
                }
            }
            code[j] = static_cast<std::uint8_t>(best);
        }
    }

    void decode(const std::uint8_t* code, float* out) const {
        for (std::size_t j = 0; j < m_; ++j) std::copy_n(centroid(j, code[j]), ds_, out + j * ds_);
    }

    // Squared distance from each sub-vector of q to each of its 256 centroids: m x 256.
    void table(const float* q, float* t) const {
        for (std::size_t j = 0; j < m_; ++j)
            for (std::size_t c = 0; c < ks; ++c) t[j * ks + c] = l2sq(centroid(j, c), q + j * ds_, ds_);
    }

    // Asymmetric distance: the query exactly, the vector through its code.
    static float adc(const float* t, const std::uint8_t* code, std::size_t m) {
        float s = 0.0f;
        for (std::size_t j = 0; j < m; ++j) s += t[j * ks + code[j]];
        return s;
    }

    std::size_t m() const { return m_; }

private:
    const float* centroid(std::size_t j, std::size_t c) const { return book_.data() + (j * ks + c) * ds_; }
    std::size_t m_, ds_;
    std::vector<float> book_;
};

class IVFPQ {
public:
    IVFPQ(const Matrix& base, std::size_t nlist, std::size_t m, std::uint64_t seed, std::size_t train = 200000,
          int iters = 20)
        : base_(base), d_(base.d), centroids_(kmeans(base, nlist, iters, seed, train).centroids), pq_(base.d, m),
          ids_(nlist), codes_(nlist) {
        // PQ learns the residuals: a vector minus its list's centroid
        std::mt19937_64 rng(seed + 1);
        std::vector<std::size_t> rows(base.n);
        std::iota(rows.begin(), rows.end(), std::size_t{0});
        std::shuffle(rows.begin(), rows.end(), rng);
        rows.resize(std::min(train, base.n));
        Matrix res(rows.size(), d_);
        for (std::size_t i = 0; i < rows.size(); ++i) residual(base.row(rows[i]), nearest(centroids_, base.row(rows[i])), res.row(i));
        pq_.train(res, seed + 2, iters);
        std::vector<float> r(d_);
        std::vector<std::uint8_t> code(m);
        for (std::size_t i = 0; i < base.n; ++i) {
            std::size_t l = nearest(centroids_, base.row(i));
            residual(base.row(i), l, r.data());
            pq_.encode(r.data(), code.data());
            ids_[l].push_back(static_cast<id_t>(i));
            codes_[l].insert(codes_[l].end(), code.begin(), code.end());
        }
    }

    // `rerank` > 0: keep that many candidates by compressed distance, then order them by
    // exact distance (needs the original vectors, which this index keeps a reference to).
    std::vector<Hit> search(const float* q, std::size_t k, std::size_t nprobe, std::size_t rerank = 0) const {
        const std::size_t m = pq_.m();
        TopK top(rerank ? std::max(rerank, k) : k);
        std::vector<float> r(d_), t(m * PQ::ks);
        for (std::size_t l : probe(centroids_, q, nprobe)) {
            residual(q, l, r.data());
            pq_.table(r.data(), t.data());
            const std::uint8_t* c = codes_[l].data();
            for (std::size_t i = 0; i < ids_[l].size(); ++i) top.push(PQ::adc(t.data(), c + i * m, m), ids_[l][i]);
        }
        std::vector<Hit> hits = top.sorted();
        if (!rerank) return hits;
        TopK exact(k);
        for (const Hit& h : hits) exact.push(l2sq(base_.row(h.id), q, d_), h.id);
        return exact.sorted();
    }

    std::size_t bytes() const {  // the index only; reranking also reads the original vectors
        std::size_t b = centroids_.v.size() * sizeof(float) + pq_.m() * PQ::ks * (d_ / pq_.m()) * sizeof(float);
        for (std::size_t l = 0; l < ids_.size(); ++l) b += ids_[l].size() * (sizeof(id_t) + pq_.m());
        return b;
    }

private:
    void residual(const float* x, std::size_t l, float* out) const {
        const float* c = centroids_.row(l);
        for (std::size_t t = 0; t < d_; ++t) out[t] = x[t] - c[t];
    }
    const Matrix& base_;
    std::size_t d_;
    Matrix centroids_;
    PQ pq_;
    std::vector<std::vector<id_t>> ids_;
    std::vector<std::vector<std::uint8_t>> codes_;
};

}  // namespace vs
