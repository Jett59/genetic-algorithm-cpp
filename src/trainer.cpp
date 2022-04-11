#include "trainer.h"
#include "neurons.h"
#include <cstddef>
#include <vector>

using std::vector;

namespace neurons {
static constexpr const size_t POPULATION_SIZE = 1024;

ScoredNetwork Trainer::scoreNetwork(const Network &network) {
  double score = 0;
  for (const Input &input : dataset) {
    score += lossFunction(
        input, network.apply(input.getNetworkInputs(), activationFunction));
  }
  score /= dataset.size();
  return ScoredNetwork(score, network);
}
void Trainer::train(unsigned epochs) {
  for (unsigned epoch = 0; epoch < epochs; epoch++) {
  }
}
} // namespace neurons
