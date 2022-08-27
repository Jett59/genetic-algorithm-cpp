#ifndef GENETIC_INPUT_H
#define GENETIC_INPUT_H

#include "neurons.h"
#include <string>

namespace genetic {
#define INPUT_LENGTH 20
struct Input {
  std::string word;
  NetworkInputs<INPUT_LENGTH> networkInputs;
  bool correctlySpelled;
};

static inline double score(const Input &input, double networkOutput) {
  if (input.correctlySpelled) {
    return 1 - networkOutput;
  } else {
    return networkOutput;
  }
}
} // namespace genetic

#endif