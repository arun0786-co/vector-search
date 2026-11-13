// Vectors, distances, a bounded top-k, and the file format the benchmark reads.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vs {

using id_t = std::uint32_t;

// n rows of d floats, stored row after row.
struct Matrix {
    std::size_t n = 0, d = 0;
    std::vector<float> v;

    Matrix() = default;
    Matrix(std::size_t rows, std::size_t dims) : n(rows), d(dims), v(rows * dims) {}
    const float* row(std::size_t i) const { return v.data() + i * d; }
    float* row(std::size_t i) { return v.data() + i * d; }
};

// Squared Euclidean distance. On unit vectors, ordering by this is ordering by cosine.
inline float l2sq(const float* a, const float* b, std::size_t d) {
    float s = 0.0f;
    for (std::size_t i = 0; i < d; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

struct Hit {
    float dist;
    id_t id;
    bool operator<(const Hit& o) const { return dist < o.dist || (dist == o.dist && id < o.id); }
    bool operator==(const Hit& o) const = default;
};

// The k smallest hits seen so far: a max-heap whose top is the current k-th best.
class TopK {
public:
    explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k + 1); }

    float worst() const { return heap_.size() < k_ ? std::numeric_limits<float>::infinity() : heap_.front().dist; }

    void push(float dist, id_t id) {
        Hit h{dist, id};
        if (heap_.size() < k_) {
            heap_.push_back(h);
            std::push_heap(heap_.begin(), heap_.end());
        } else if (h < heap_.front()) {
            std::pop_heap(heap_.begin(), heap_.end());
            heap_.back() = h;
            std::push_heap(heap_.begin(), heap_.end());
        }
    }

    std::vector<Hit> sorted() const {
        std::vector<Hit> out = heap_;
        std::sort(out.begin(), out.end());
        return out;
    }

private:
    std::size_t k_;
    std::vector<Hit> heap_;
};

// Index of the nearest row of `c` to `x`.
inline std::size_t nearest(const Matrix& c, const float* x) {
    std::size_t best = 0;
    float bd = std::numeric_limits<float>::infinity();
    for (std::size_t j = 0; j < c.n; ++j) {
        float dist = l2sq(c.row(j), x, c.d);
        if (dist < bd) {
            bd = dist;
            best = j;
        }
    }
    return best;
}

// Binary files: int32 rows, int32 columns, then row-major values.
inline Matrix load_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::int32_t hdr[2];
    f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    Matrix m(static_cast<std::size_t>(hdr[0]), static_cast<std::size_t>(hdr[1]));
    f.read(reinterpret_cast<char*>(m.v.data()), static_cast<std::streamsize>(m.v.size() * sizeof(float)));
    if (!f) throw std::runtime_error("short read in " + path);
    return m;
}

inline std::vector<std::int32_t> load_i32(const std::string& path, std::size_t& rows, std::size_t& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::int32_t hdr[2];
    f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    rows = static_cast<std::size_t>(hdr[0]);
    cols = static_cast<std::size_t>(hdr[1]);
    std::vector<std::int32_t> out(rows * cols);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(std::int32_t)));
    if (!f) throw std::runtime_error("short read in " + path);
    return out;
}

// Exact search: every distance.
inline std::vector<Hit> exact_search(const Matrix& base, const float* q, std::size_t k) {
    TopK top(k);
    for (std::size_t i = 0; i < base.n; ++i) top.push(l2sq(base.row(i), q, base.d), static_cast<id_t>(i));
    return top.sorted();
}

}  // namespace vs
