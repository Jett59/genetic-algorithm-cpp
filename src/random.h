#ifndef _RANDOM_H
#define _RANDOM_H

#include <cstddef>
#include <random>

namespace random {
class Random {
private:
  static std::random_device seedGenerator;

  std::ranlux24_base engine;
  std::uniform_int_distribution<uint_fast64_t> intDistribution;
  std::uniform_real_distribution<double> realDistribution;

public:
  Random() : engine(seedGenerator()) {}

  uint_fast64_t getInt(uint_fast64_t range) { return intDistribution(engine) % range; }
  double getDouble() { return realDistribution(engine); }
};
} // namespace random

#endif