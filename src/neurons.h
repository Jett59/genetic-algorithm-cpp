#ifndef GENETIC_NEURONS_H
#define GENETIC_NEURONS_H

#include "random.h"
#include <array>
#include <cstddef>

namespace genetic {
using ActivationFunction = double (*)(double);

template <size_t len> using NetworkInputs = std::array<double, len>;

template <size_t weightCount> struct Neuron {
  double weights[weightCount];
  double bias;
};
template <ActivationFunction activationFunction, size_t neuronCount,
          size_t nextLayerNeuronCount>
struct Layer {
  Neuron<nextLayerNeuronCount> neurons[neuronCount];

  template <typename Random>
  void combineWith(Random &rand, const Layer<activationFunction, neuronCount,
                                             nextLayerNeuronCount> &other) {
    for (size_t i = 0; i < neuronCount; i++) {
      for (size_t j = 0; j < neuronCount; j++) {
        neurons[i].weights[j] =
            rand() < 0.5 ? neurons[i].weights[j] : other.neurons[i].weights[j];
      }
      neurons[i].bias = rand() < 0.5 ? neurons[i].bias : other.neurons[i].bias;
    }
  }

  template <typename Random> void randomize(Random &rand) {
    for (auto &neuron : neurons) {
      for (auto &weight : neuron.weights) {
        weight = rand() * 2 - 1;
      }
      neuron.bias = rand() * 2 - 1;
    }
  }

  NetworkInputs<nextLayerNeuronCount>
  apply(const NetworkInputs<neuronCount> &inputs) const {
    NetworkInputs<nextLayerNeuronCount> result;
    std::fill(result.begin(), result.end(), 0);
    for (size_t neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++) {
      const Neuron<nextLayerNeuronCount> &neuron = neurons[neuronIndex];
      const double neuronValue = inputs[neuronIndex];
      for (size_t weightIndex = 0; weightIndex < nextLayerNeuronCount;
           weightIndex++) {
        result[weightIndex] +=
            neuronValue * neuron.weights[weightIndex] + neuron.bias;
      }
    }
    for (auto &resultValue : result) {
      resultValue = activationFunction(resultValue);
    }
    return result;
  }

  template <typename Random> void mutate(Random &rand, double mutationRate) {
    for (auto &neuron : neurons) {
      for (auto &weight : neuron.weights) {
        if (rand() < mutationRate) [[unlikely]] {
          weight *= rand() * 4 - 2;
        }
      }
      if (rand() < mutationRate) [[unlikely]] {
        neuron.bias *= rand() * 4 - 2;
      }
    }
  }
};
template <ActivationFunction activationFunction, size_t...> struct Network;
template <ActivationFunction activationFunction, size_t currentLayerSize>
struct Network<activationFunction, currentLayerSize> {
  template <typename Random>
  void combineWith(Random &,
                   const Network<activationFunction, currentLayerSize> &) {}
  template <typename Random> void randomize(Random &) {}
  template <typename Random> void mutate(Random &, double) {}

  auto apply(const NetworkInputs<currentLayerSize> &inputs) const {
    return inputs;
  }
};
template <ActivationFunction activationFunction, size_t currentLayerSize,
          size_t nextLayerSize, size_t... otherLayerSizes>
struct Network<activationFunction, currentLayerSize, nextLayerSize,
               otherLayerSizes...> {
  Layer<activationFunction, currentLayerSize, nextLayerSize> layer;
  Network<activationFunction, nextLayerSize, otherLayerSizes...> nextLayers;

  template <typename Random>
  void
  combineWith(Random &rand,
              const Network<activationFunction, currentLayerSize, nextLayerSize,
                            otherLayerSizes...> &otherNetwork) {
    layer.combineWith(rand, otherNetwork.layer);
    nextLayers.combineWith(rand, otherNetwork.nextLayers);
  }

  template <typename Random> void randomize(Random &rand) {
    layer.randomize(rand);
    nextLayers.randomize(rand);
  }
  template <typename Random> void mutate(Random &rand, double mutationRate) {
    layer.mutate(rand, mutationRate);
    nextLayers.mutate(rand, mutationRate);
  }

  auto apply(const NetworkInputs<currentLayerSize> &inputs) const {
    return nextLayers.apply(layer.apply(inputs));
  }
};
} // namespace genetic

// Read and write the networks.
template <genetic::ActivationFunction activationFunction, size_t... layerSizes>
std::ostream &
operator<<(std::ostream &os,
           const genetic::Network<activationFunction, layerSizes...> &network) {
  os << network.layer << network.nextLayers;
  return os;
}
template <genetic::ActivationFunction activationFunction, size_t lastLayerSize>
std::ostream &
operator<<(std::ostream &os,
           const genetic::Network<activationFunction, lastLayerSize> &network) {
  return os;
}
template <genetic::ActivationFunction activationFunction, size_t... layerSizes>
std::istream &
operator>>(std::istream &is,
           genetic::Network<activationFunction, layerSizes...> &network) {
  is >> network.layer >> network.nextLayers;
  return is;
}
template <genetic::ActivationFunction activationFunction, size_t lastLayerSize>
std::istream &
operator>>(std::istream &is,
           genetic::Network<activationFunction, lastLayerSize> &network) {
  return is;
}

template <genetic::ActivationFunction activationFunction, size_t layerSize,
          size_t nextLayerSize>
std::ostream &operator<<(
    std::ostream &os,
    const genetic::Layer<activationFunction, layerSize, nextLayerSize> &layer) {
  for (const auto &neuron : layer.neurons) {
    for (const auto &weight : neuron.weights) {
      os.write(reinterpret_cast<const char *>(&weight), sizeof(weight));
    }
    os.write(reinterpret_cast<const char *>(&neuron.bias), sizeof(neuron.bias));
  }
  return os;
}
template <genetic::ActivationFunction activationFunction, size_t layerSize,
          size_t nextLayerSize>
std::istream &operator>>(
    std::istream &is,
    genetic::Layer<activationFunction, layerSize, nextLayerSize> &layer) {
  for (auto &neuron : layer.neurons) {
    for (auto &weight : neuron.weights) {
      is.read(reinterpret_cast<char *>(&weight), sizeof(weight));
    }
    is.read(reinterpret_cast<char *>(&neuron.bias), sizeof(neuron.bias));
  }
  return is;
}

#endif