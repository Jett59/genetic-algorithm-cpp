#ifndef GENETIC_NEURONS_H
#define GENETIC_NEURONS_H

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

  auto apply(const NetworkInputs<neuronCount> &inputs) const {
    NetworkInputs<nextLayerNeuronCount> result;
    for (size_t neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++) {
    const Neuron<nextLayerNeuronCount>   &neuron = neurons[neuronIndex];
      const double neuronValue = inputs[neuronIndex];
      for (size_t weightIndex = 0; weightIndex < nextLayerNeuronCount;
           weightIndex++) {
        result[weightIndex] +=
            neuronValue * neuron.weights[weightIndex] + neuron.bias;
      }
    }
    for (size_t resultIndex = 0; resultIndex < nextLayerNeuronCount;
         resultIndex++) {
      result[resultIndex] = activationFunction(result[resultIndex]);
    }
    return result;
  }
};
template <ActivationFunction activationFunction, size_t...> struct Network;
template <ActivationFunction activationFunction, size_t currentLayerSize>
struct Network<activationFunction, currentLayerSize> {
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

  auto apply(const NetworkInputs<currentLayerSize> &inputs) const {
    return nextLayers.apply(layer.apply(inputs));
  }
};
} // namespace genetic

#endif