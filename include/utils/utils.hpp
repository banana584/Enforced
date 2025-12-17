#ifndef ENFORCED_UTILS_HEADER
#define ENFORCED_UTILS_HEADER

#include <random>

static std::random_device rd;
static std::mt19937 gen(rd());

template <typename T>
T GetRandomNumber(T lower, T higher) {
    std::uniform_real_distribution<T> dist(lower, higher);

    return dist(gen);
}

#endif