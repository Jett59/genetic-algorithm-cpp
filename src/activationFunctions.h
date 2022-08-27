#ifndef GENETIC_ACTIVATION_FUNCTIONS_H
#define GENETIC_ACTIVATION_FUNCTIONS_H

#include "neurons.h"
#include <cmath>

#include <iostream>

namespace genetic {
static constexpr double DEFAULT_ACTIVATION(double x) {
  return (x / (abs(x) + 1) + 1) / 2;
};
}

#endif