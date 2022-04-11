#ifndef NEURON_TRAINER_H
#define NEURON_TRAINER_H

#include "neurons.h"
#include <any>
#include <functional>
#include <set>
#include <utility>

namespace neurons {
class ScoredNetwork {
private:
  const double score;
  const Network network;

public:
  ScoredNetwork(double score, const Network network)
      : score(score), network(std::move(network)) {}

  double getScore() const { return score; }
  const Network &getNetwork() const { return network; }

  bool operator<(const ScoredNetwork &other) const {
    return score < other.score;
  }
  bool operator>(const ScoredNetwork &other) const {
    return score > other.score;
  }
};

class Input {
private:
  const std::vector<double> networkInputs;
  const std::any data;

public:
  Input(const std::vector<double> &networkInputs, const std::any &data)
      : networkInputs(networkInputs), data(data) {}

  const std::vector<double> &getNetworkInputs() const { return networkInputs; }

  const std::any &getData() const { return data; }
};

using LossFunction =
    const std::function<double(const Input &, const std::vector<double> &)>;

class Trainer {
private:
  std::set<ScoredNetwork> networks;
  const std::vector<Input> &dataset;
  const LossFunction &lossFunction;
  const ActivationFunction &activationFunction;

  ScoredNetwork scoreNetwork(const Network &network);

public:
  Trainer(const std::vector<Input> &dataset, const LossFunction &lossFunction, const ActivationFunction &activationFunction)
      : dataset(dataset), lossFunction(lossFunction), activationFunction(activationFunction) {}

  const Network &getBestNetwork() { return networks.begin()->getNetwork(); }
  double getBestScore() { return networks.begin()->getScore(); }

  void train(unsigned epochs);
};
} // namespace neurons

#endif