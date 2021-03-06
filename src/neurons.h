#ifndef NEURONS_NEURONS_H
#define NEURONS_NEURONS_H

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

  Neuron generateNeuron(Random &rand, size_t weightCount) const;
  Layer generateLayer(Random &rand, size_t layerSize,
                      size_t nextLayerSize) const;

public:
  NetworkDescription(std::vector<size_t> layers)
      : layerDescriptions(std::move(layers)) {}

  size_t layerSize(size_t index) const { return layerDescriptions[index]; }
  size_t layerCount() const { return layerDescriptions.size(); }

  Network generateNetwork(Random &rand) const;
};

class Neuron {
private:
  std::vector<double> weights;
  std::vector<double> biases;

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

  Neuron mutate(unsigned mutationRate, Random &rand) const;
};
class Layer {
private:
  std::vector<Neuron> neurons;

public:
  Layer(const std::vector<Neuron> neurons) : neurons(std::move(neurons)) {}

  const Neuron &getNeuron(size_t index) const { return neurons[index]; }
  size_t neuronCount() const { return neurons.size(); }

  // Returns the values for the next layer
  std::vector<double> apply(const std::vector<double> &input,
                            ActivationFunction &activationFunction,
                            size_t nextLayerSize) const;

  Layer mutate(unsigned mutationRate, Random &rand) const;
};
class Network {
private:
  NetworkDescription description;
  std::vector<Layer> layers;

public:
  Network(const NetworkDescription description, const std::vector<Layer> layers)
      : description(std::move(description)), layers(std::move(layers)) {}

  const Layer &getLayer(size_t index) const { return layers[index]; }
  size_t layerCount() const { return description.layerCount(); }

  size_t layerSize(size_t index) const { return description.layerSize(index); }

  // Returns the result of applying the inputs
  std::vector<double> apply(const std::vector<double> &inputs,
                            ActivationFunction &activationFunction) const;

  Network mutate(unsigned mutationRate, Random &rand) const;
};
} // namespace neurons

#endif