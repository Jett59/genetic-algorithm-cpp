#include "neurons.h"
#include <cstddef>
#include <utility>
#include <vector>

using std::move;
using std::vector;

namespace neurons {
vector<double> Layer::apply(const vector<double> &inputs,
                            ActivationFunction &activationFunction,
                            const Layer &nextLayer) const {
  vector<double> result(nextLayer.neuronCount());
  for (size_t i = 0; i < nextLayer.neuronCount(); i++) {
    double value = 0;
    for (size_t j = 0; j < neuronCount(); i++) {
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
    temp = layers[i].apply(temp, activationFunction, layers[i + 1]);
  }
  return temp;
}
} // namespace neurons
