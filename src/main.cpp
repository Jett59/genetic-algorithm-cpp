#include "activationFunctions.h"
#include "neurons.h"
#include <iostream>

using namespace genetic;

int main() {
  Network<DEFAULT_ACTIVATION, 20, 30, 1> network;
  network.apply(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  return 0;
}
