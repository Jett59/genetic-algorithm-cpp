#ifndef _NEURONS_H
#define _NEURONS_H

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neurons {
using ActivationFunction = const std::function<double(double)>;

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
                            const Layer &nextLayer) const;
};
class Network {
private:
  const std::vector<Layer> layers;

public:
  Network(const std::vector<Layer> layers) : layers(std::move(layers)) {}

  const Layer &getLayer(size_t index) const { return layers[index]; }
  size_t layerCount() const { return layers.size(); }

  // Returns the result of applying the inputs
  std::vector<double> apply(const std::vector<double> &inputs,
                            ActivationFunction &activationFunction) const;
};
} // namespace neurons

#endif