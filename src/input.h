#ifndef GENETIC_INPUT_H
#define GENETIC_INPUT_H

#include "neurons.h"
#include <string>

namespace genetic {
#define INPUT_LENGTH 20
struct Input {
  NetworkInputs<INPUT_LENGTH> networkInputs;
  bool correctlySpelled;

  Input(const std::string &word, bool correctlySpelled)
      : correctlySpelled(correctlySpelled) {
    for (size_t i = 0; i < INPUT_LENGTH; i++) {
      networkInputs[i] = word[i];
    }
  }
};

static inline double score(const Input &input,
                           const NetworkInputs<1> &networkOutputs) {
  double networkOutput = networkOutputs[0];
  if (input.correctlySpelled) {
    return 1 - networkOutput;
  } else {
    return networkOutput;
  }
}
} // namespace genetic

#endif