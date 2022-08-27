#include "activationFunctions.h"
#include "neurons.h"
#include <iostream>

using namespace genetic;

int main() {
  DefaultRandom rand;
  Network<DEFAULT_ACTIVATION, 20, 30, 1> network;
  network.randomize(rand);
  auto result = network.apply(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  for (auto value : result) {
    std::cout << value << std::endl;
  }
  return 0;
}
