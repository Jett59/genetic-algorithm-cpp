#include "neurons.h"
#include "trainer.h"
#include "random.h"
#include <iostream>
#include <vector>

using neurons::ActivationFunction;
using neurons::Network;
using neurons::NetworkDescription;
using neurons::Random;
using std::cout;
using std::endl;
using std::random_device;
using std::vector;

random_device Random::seedGenerator;

int main() {
  cout << "Genetic algorithm" << endl;
  NetworkDescription description({2, 5, 1});
  Random rand;
  Network network = description.generateNetwork(rand);
  ActivationFunction activationFunction = [](double input) -> double {
    return input / 2;
  };
  const vector<double> networkInputs = {2, 5};
  vector<double> networkOutputs =
      network.apply(networkInputs, activationFunction);
  cout << "Output:" << endl;
  for (double output : networkOutputs) {
    cout << output << endl;
  }
  return 0;
}
