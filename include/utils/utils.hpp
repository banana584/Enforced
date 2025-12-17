#ifndef ENFORCED_UTILS_HEADER
#define ENFORCED_UTILS_HEADER

#include <random>

// Random device and generator for random numbers - large classes so we only have 1 and keep it here
static std::random_device rd;
static std::mt19937 gen(rd());

/**
 * @brief Generates a random number within a range
 * 
 * @warning The template should only be types that are numbers - e.g int, float, double
 * 
 * @param lower The lower boundary of the random number
 * @param higher The upper boundary of the random number
 * 
 * @return A number of the type specified in the template
 */
template <typename T>
T GetRandomNumber(T lower, T higher) {
    std::uniform_real_distribution<T> dist(lower, higher);

    return dist(gen);
}

#endif