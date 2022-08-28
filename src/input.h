#ifndef GENETIC_INPUT_H
#define GENETIC_INPUT_H

#include "neurons.h"
#include <string>
#include <atomic>
namespace genetic {
#define INPUT_LENGTH 20
struct Input {
  std::string word;
  NetworkInputs<INPUT_LENGTH> networkInputs;
  bool correctlySpelled;

  Input(const std::string &word, bool correctlySpelled)
      : correctlySpelled(correctlySpelled), word(word) {
    for (size_t i = 0; i < INPUT_LENGTH; i++) {
      networkInputs[i] = word[i];
    }
  }
};
static std::atomic_int correctlyVsNotCount = 0;
static inline double score(const Input &input,
                           const NetworkInputs<1> &networkOutputs) {
  double networkOutput = networkOutputs[0];
  if (input.correctlySpelled) {
    correctlyVsNotCount++;
    return networkOutput;
  } else {
    correctlyVsNotCount--;
    return 1-networkOutput;
  }
}
} // namespace genetic

#endif