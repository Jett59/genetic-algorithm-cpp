#ifndef GENETIC_TRAINER_H
#define GENETIC_TRAINER_H

#include "neurons.h"
#include "random.h"
#include <deque>

namespace genetic {
template <typename Network> struct ScoredNetwork {
  double score;
  Network network;

  bool operator<(const ScoredNetwork &other) const {
    return score < other.score;
  }
  bool operator>(const ScoredNetwork &other) const {
    return score > other.score;
  }
  bool operator==(const ScoredNetwork &other) const {
    return score == other.score;
  }
};

template <typename Network, typename Random, typename Input,
          double (*scorer)(const Input &,
                           std::add_lvalue_reference_t<std::add_const_t<
                               decltype(std::declval<Network>().apply(
                                   std::declval<Input>().networkInputs))>>),
          typename InputIterator, size_t populationSize>
class Trainer {
  std::deque<ScoredNetwork<Network>> networks;
  Random rand;
  InputIterator begin, end;

public:
  Trainer(Random rand, InputIterator begin, InputIterator end)
      : rand(rand), begin(begin), end(end) {
    for (size_t i = 0; i < populationSize; i++) {
      Network network;
      network.randomize(rand);
      networks.push_back(score(network));
    }
    std::sort(networks.begin(), networks.end());
  }

  void train(size_t iterations, size_t mutationRate) {
    for (size_t i = 0; i < iterations; i++) {
      for (size_t j = 0; j < populationSize / 2; j++) {
        networks.pop_front();
      }
      for (size_t j = 0; j < populationSize / 2; j++) {
        Network newNetwork = networks[j].network;
        newNetwork.randomize(rand);
        networks.push_back(score(newNetwork));
      }
      std::sort(networks.begin(), networks.end());
    }
  }

  ScoredNetwork<Network> score(const Network &network) const {
    ScoredNetwork<Network> scoredNetwork;
    scoredNetwork.network = network;
    double score = 0;
    for (auto input = begin; input != end; input++) {
      score += scorer(*input, network.apply(input->networkInputs));
    }
    scoredNetwork.score = score / std::distance(begin, end);
    return scoredNetwork;
  }

  const ScoredNetwork<Network> &best() const { return networks.back(); }
};
} // namespace genetic

#endif