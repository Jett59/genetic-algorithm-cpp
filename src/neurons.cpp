#include "neurons.h"
#include "random.h"
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

using std::invalid_argument;
using std::move;
using std::vector;

namespace neurons {
vector<double> Layer::apply(const vector<double> &inputs,
                            ActivationFunction &activationFunction,
                            size_t nextLayerSize) const {
  vector<double> result;
  result.reserve(nextLayerSize);
  for (size_t i = 0; i < nextLayerSize; i++) {
    double value = 0;
    for (size_t j = 0; j < neuronCount(); j++) {
      const Neuron &neuron = neurons[j];
      value += neuron.apply(inputs[j], i);
    }
    result.push_back(activationFunction(value));
  }
  return result;
}
vector<double> Network::apply(const vector<double> &inputs,
                              ActivationFunction &activationFunction) const {
  vector<double> temp = inputs;
  for (size_t i = 0; i < layerCount() - 1; i++) {
    if (temp.size() != layers[i].neuronCount()) {
      throw invalid_argument("Wrong number of arguments to neural layer");
    }
    temp =
        layers[i].apply(temp, activationFunction, description.layerSize(i + 1));
  }
  return temp;
}

Neuron NetworkDescription::generateNeuron(Random &rand,
                                          size_t weightCount) const {
  vector<double> weights;
  weights.reserve(weightCount);
  vector<double> biases;
  biases.reserve(weightCount);
  for (size_t i = 0; i < weightCount; i++) {
    weights.push_back(rand.getDouble());
    biases.push_back(rand.getDouble());
  }
  return Neuron(weights, biases);
}
Layer NetworkDescription::generateLayer(Random &rand, size_t layerSize,
                                        size_t nextLayerSize) const {
  vector<Neuron> neurons;
  neurons.reserve(layerSize);
  for (size_t i = 0; i < layerSize; i++) {
    neurons.push_back(generateNeuron(rand, nextLayerSize));
  }
  return Layer(neurons);
}
Network NetworkDescription::generateNetwork(Random &rand) const {
  size_t layerCount = layerDescriptions.size();
  vector<Layer> layers;
  layers.reserve(layerCount - 1);
  for (size_t i = 0; i < layerCount - 1; i++) {
    layers.push_back(
        generateLayer(rand, layerDescriptions[i], layerDescriptions[i + 1]));
  }
  return Network(*this, layers);
}
} // namespace neurons
