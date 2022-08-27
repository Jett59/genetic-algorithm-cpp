#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "activationFunctions.h"
#include "input.h"
#include "neurons.h"
#include "trainer.h"

using namespace genetic;

static std::vector<Input> readInputs(const std::string &fileName) {
  std::vector<Input> inputs;
  std::ifstream file(fileName);
  if (!file) {
    std::cerr << "Could not open file " << fileName << std::endl;
    exit(-1);
  }
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream lineStream(line);
    std::string word;
    lineStream >> word;
    bool correctlySpelled;
    lineStream >> correctlySpelled;
    inputs.push_back({word, correctlySpelled});
  }
  return inputs;
}

using DefaultNetwork = Network<DEFAULT_ACTIVATION, INPUT_LENGTH, 25, 1>;
using DefaultTrainer = Trainer<DefaultNetwork, DefaultRandom, Input, &score,
                               std::vector<Input>::iterator, 1024>;

int main() {
  DefaultRandom rand;
  std::vector<Input> inputs = readInputs("inputs.txt");
  DefaultTrainer trainer(rand, inputs.begin(), inputs.end());
  std::cout << trainer.best().score << std::endl;
  return 0;
}
