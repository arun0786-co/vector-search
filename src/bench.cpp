// Build an index on the base vectors, then answer every query at each search setting and
// report recall@10 against the published true neighbours, queries per second (one thread),
// build time and index size. One CSV line per search setting.
//
//   bench DATA exact [queries]
//   bench DATA ivf    NLIST            NPROBE,NPROBE,...
//   bench DATA ivfpq  NLIST M RERANK   NPROBE,NPROBE,...
//   bench DATA hnsw   M EF_CONSTRUCTION EF,EF,...
#include <chrono>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>

#include "vs/hnsw.hpp"
#include "vs/ivf.hpp"

using namespace vs;
using Clock = std::chrono::steady_clock;

static double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

static std::vector<std::size_t> list(const std::string& s) {
    std::vector<std::size_t> out;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) out.push_back(std::stoul(item));
    return out;
}

struct Data {
    Matrix base, queries;
    std::vector<std::int32_t> gt;
    std::size_t gt_cols = 0;
};

// Recall@10 and queries per second for one search setting.
static void measure(const Data& d, std::size_t nq, const std::string& index, const std::string& params,
                    double build_s, std::size_t bytes, const std::string& setting,
                    const std::function<std::vector<Hit>(const float*)>& search) {
    double found = 0;
    auto t = Clock::now();
    std::vector<std::vector<Hit>> results(nq);
    for (std::size_t i = 0; i < nq; ++i) results[i] = search(d.queries.row(i));
    double secs = seconds_since(t);
    for (std::size_t i = 0; i < nq; ++i) {
        std::set<std::int32_t> truth(d.gt.begin() + static_cast<std::ptrdiff_t>(i * d.gt_cols),
                                     d.gt.begin() + static_cast<std::ptrdiff_t>(i * d.gt_cols + 10));
        for (const Hit& h : results[i]) found += static_cast<double>(truth.count(static_cast<std::int32_t>(h.id)));
    }
    std::printf("%s,%s,%.1f,%.1f,%s,%.0f,%.4f\n", index.c_str(), params.c_str(), build_s,
                static_cast<double>(bytes) / 1e6, setting.c_str(), static_cast<double>(nq) / secs,
                found / (10.0 * static_cast<double>(nq)));
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: bench DATA exact|ivf|ivfpq|hnsw ...\n";
        return 2;
    }
    std::string dir = argv[1], kind = argv[2];
    Data d;
    d.base = load_f32(dir + "/train.f32");
    d.queries = load_f32(dir + "/test.f32");
    std::size_t gt_rows = 0;
    d.gt = load_i32(dir + "/gt.i32", gt_rows, d.gt_cols);
    const std::size_t nq = d.queries.n;
    const std::size_t raw = d.base.v.size() * sizeof(float);

    if (kind == "exact") {
        std::size_t n = argc > 3 ? std::stoul(argv[3]) : 1000;
        measure(d, std::min(n, nq), "exact", "-", 0.0, raw, "-",
                [&](const float* q) { return exact_search(d.base, q, 10); });
    } else if (kind == "ivf" && argc == 5) {
        std::size_t nlist = std::stoul(argv[3]);
        auto t = Clock::now();
        IVFFlat ix(d.base, nlist, 1);
        double b = seconds_since(t);
        for (std::size_t p : list(argv[4]))
            measure(d, nq, "ivf", "nlist=" + std::to_string(nlist), b, ix.bytes(), "nprobe=" + std::to_string(p),
                    [&](const float* q) { return ix.search(q, 10, p); });
    } else if (kind == "ivfpq" && argc == 7) {
        std::size_t nlist = std::stoul(argv[3]), m = std::stoul(argv[4]), rerank = std::stoul(argv[5]);
        auto t = Clock::now();
        IVFPQ ix(d.base, nlist, m, 1);
        double b = seconds_since(t);
        std::string params = "nlist=" + std::to_string(nlist) + " m=" + std::to_string(m) + " rerank=" + std::to_string(rerank);
        for (std::size_t p : list(argv[6]))
            measure(d, nq, "ivfpq", params, b, ix.bytes(), "nprobe=" + std::to_string(p),
                    [&](const float* q) { return ix.search(q, 10, p, rerank); });
    } else if (kind == "hnsw" && argc == 6) {
        std::size_t M = std::stoul(argv[3]), efc = std::stoul(argv[4]);
        auto t = Clock::now();
        HNSW ix(d.base, M, efc, 1);
        double b = seconds_since(t);
        std::string params = "M=" + std::to_string(M) + " efC=" + std::to_string(efc);
        for (std::size_t ef : list(argv[5]))
            measure(d, nq, "hnsw", params, b, ix.bytes() + raw, "ef=" + std::to_string(ef),
                    [&](const float* q) { return ix.search(q, 10, ef); });
    } else {
        std::cerr << "bad arguments\n";
        return 2;
    }
    return 0;
}
