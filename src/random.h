#ifndef GENETIC_RANDOM_H
#define GENETIC_RANDOM_H

#include <random>

namespace genetic {
template <typename RandomEngine> class Random {
  RandomEngine engine;

public:
  Random(std::random_device &seedGenerator) : engine(seedGenerator()) {}
  Random() : engine(std::random_device()()) {}

  double operator()() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(engine);
  }
};
using DefaultRandom = Random<std::ranlux24_base>;
} // namespace genetic

#endif