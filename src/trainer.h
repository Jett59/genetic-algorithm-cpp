#ifndef GENETIC_TRAINER_H
#define GENETIC_TRAINER_H

#include "neurons.h"
#include "random.h"
#include "workers.h"
#include <algorithm>
#include <vector>

namespace genetic {
template <typename Network> struct ScoredNetwork {
  double score;
  double age;
  Network network;

  double rank() const { return score / age; }
  bool operator<(const ScoredNetwork &other) const {
    return rank() < other.rank();
  }
  bool operator>(const ScoredNetwork &other) const {
    return rank() > other.rank();
  }
  bool operator==(const ScoredNetwork &other) const {
    return rank() == other.rank();
  }
};

template <typename Network, typename Random, typename Input,
          double (*scorer)(const Input &,
                           std::add_lvalue_reference_t<std::add_const_t<
                               decltype(std::declval<Network>().apply(
                                   std::declval<Input>().networkInputs))>>),
          typename InputIterator, size_t populationSize>
class Trainer {
  std::vector<ScoredNetwork<Network>> networks;
  Random rand;
  InputIterator begin, end;

  static double scoreNetwork(const Input &input, const Network &network) {
    return scorer(input, network.apply(input.networkInputs));
  }

  WorkerPool<Input, Network, double, scoreNetwork> workerPool;

public:
  Trainer(Random rand, InputIterator begin, InputIterator end)
      : rand(rand), begin(begin), end(end) {
    for (size_t i = 0; i < populationSize; i++) {
      Network network;
      network.randomize(rand);
      auto scoredNetwork = createScoredNetwork(std::move(network));
      workerPool(begin, end, scoredNetwork.network);
      scoredNetwork.score = workerPool.await() / std::distance(begin, end);
      networks.push_back(std::move(scoredNetwork));
    }
    std::sort(networks.begin(), networks.end());
  }

  void train(size_t iterations, double mutationRate) {
    for (size_t i = 0; i < iterations; i++) {
      for (size_t j = 0; j < populationSize / 2; j++) {
        auto &scoredNetwork = networks[j + populationSize / 2];
        scoredNetwork.age += 0.001;
        auto newNetwork = scoredNetwork.network;
        newNetwork.combineWith(rand, networks[populationSize - j % 16 - 1].network);
        newNetwork.mutate(rand, mutationRate);
        if (j > 0) {
          networks[j - 1].score =
              workerPool.await() / std::distance(begin, end);
        }
        networks[j] = createScoredNetwork(std::move(newNetwork));
        workerPool(begin, end, networks[j].network);
      }
      networks[populationSize / 2 - 1].score =
          workerPool.await() / std::distance(begin, end);
      std::sort(networks.begin(), networks.end());
    }
  }

  ScoredNetwork<Network> createScoredNetwork(Network network) {
    ScoredNetwork<Network> scoredNetwork;
    scoredNetwork.network = std::move(network);
    scoredNetwork.score = 0;
    scoredNetwork.age = 1;
    return scoredNetwork;
  }

void insertNetwork(Network network) {
  auto scoredNetwork = createScoredNetwork(std::move(network));
  workerPool(begin, end, scoredNetwork.network);
  scoredNetwork.score = workerPool.await() / std::distance(begin, end);
  networks[0] = std::move(scoredNetwork);
  std::sort(networks.begin(), networks.end());
}

  const ScoredNetwork<Network> &best() const {
    const ScoredNetwork<Network> *best = nullptr;
    for (auto &network : networks) {
      if (best == nullptr || network.score > best->score) {
        best = &network;
      }
    }
    return *best;
  }
};
} // namespace genetic

#endif