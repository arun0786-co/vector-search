CXX ?= c++
CXXFLAGS ?= -std=c++20 -O3 -Wall -Wextra -Wpedantic -Wconversion -Werror -Iinclude
HEADERS := $(wildcard include/vs/*.hpp) tests/check.hpp

all: build/test build/bench

build/test: tests/test.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $< -o $@

build/bench: src/bench.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $< -o $@

build:
	mkdir -p build

test: build/test
	./build/test

# address and undefined behaviour sanitizers over the tests
sanitize: | build
	$(CXX) $(CXXFLAGS) -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all tests/test.cpp -o build/test_san
	./build/test_san

clean:
	rm -rf build

.PHONY: all test sanitize clean
