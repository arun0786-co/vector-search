// A test harness in thirty lines, so the project needs no dependencies.
#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace check {

struct Case {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<Case>& cases() {
    static std::vector<Case> c;
    return c;
}
inline int& failures() {
    static int f = 0;
    return f;
}

struct Register {
    Register(const char* name, std::function<void()> fn) { cases().push_back({name, std::move(fn)}); }
};

inline int run_all() {
    for (auto& c : cases()) {
        int before = failures();
        c.fn();
        std::printf("%s  %s\n", failures() == before ? "ok  " : "FAIL", c.name.c_str());
    }
    std::printf("%zu tests, %d failed checks\n", cases().size(), failures());
    return failures() == 0 ? 0 : 1;
}

}  // namespace check

#define TEST(name) \
    static void test_##name(); \
    static check::Register reg_##name(#name, test_##name); \
    static void test_##name()

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            ++check::failures(); \
            std::printf("    %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
        } \
    } while (0)
