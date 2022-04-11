#ifndef _NEURONS_H
#define _NEURONS_H

#include "random.h"
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neurons {
using ActivationFunction = const std::function<double(double)>;

class Neuron;
class Layer;
class Network;

class NetworkDescription {
private:
  std::vector<size_t> layerDescriptions;

  Neuron generateNeuron(random::Random &rand, size_t weightCount) const;
  Layer generateLayer(random::Random &rand, size_t layerSize,
                      size_t nextLayerSize) const;

public:
  NetworkDescription(std::vector<size_t> layers)
      : layerDescriptions(std::move(layers)) {}

  size_t layerSize(size_t index) const { return layerDescriptions[index]; }

  Network generateNetwork(random::Random &rand) const;
};

class Neuron {
private:
  const std::vector<double> weights;
  const std::vector<double> biases;

public:
  Neuron(const std::vector<double> weights, const std::vector<double> biases)
      : weights(std::move(weights)), biases(std::move(biases)) {
    if (this->weights.size() != this->biases.size()) {
      throw std::invalid_argument("Bias and weight must be of equal length");
    }
  }

  double getWeight(size_t index) const { return weights[index]; }
  double getBias(size_t index) const { return biases[index]; }

  size_t weightCount() const { return weights.size(); }

  double apply(double input, size_t index) const {
    return input * weights[index] + biases[index];
  }
};
class Layer {
private:
  const std::vector<Neuron> neurons;

public:
  Layer(const std::vector<Neuron> neurons) : neurons(std::move(neurons)) {}

  const Neuron &getNeuron(size_t index) const { return neurons[index]; }
  size_t neuronCount() const { return neurons.size(); }

  // Returns the values for the next layer
  std::vector<double> apply(const std::vector<double> &input,
                            ActivationFunction &activationFunction,
                            size_t nextLayerSize) const;
};
class Network {
private:
  const NetworkDescription description;
  const std::vector<Layer> layers;

public:
  Network(const NetworkDescription description, const std::vector<Layer> layers)
      : description(std::move(description)), layers(std::move(layers)) {}

  const Layer &getLayer(size_t index) const { return layers[index]; }
  size_t layerCount() const { return layers.size(); }

  size_t layerSize(size_t index) const { return description.layerSize(index); }

  // Returns the result of applying the inputs
  std::vector<double> apply(const std::vector<double> &inputs,
                            ActivationFunction &activationFunction) const;
};
} // namespace neurons

#endif