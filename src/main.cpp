#include <csignal>
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
    std::string correctlySpelled;
    lineStream >> correctlySpelled;
    inputs.push_back({word, correctlySpelled == "true"});
  }
  return inputs;
}

using DefaultNetwork = Network<DEFAULT_ACTIVATION, INPUT_LENGTH, 50, 50, 50, 1>;
using DefaultTrainer = Trainer<DefaultNetwork, DefaultRandom, Input, &score,
                               std::vector<Input>::iterator, 512>;

int main() {
  DefaultRandom rand;
  std::vector<Input> inputs = readInputs("inputs.txt");
  std::cout << "Initializing the trainer..." << std::endl;
  std::unique_ptr<DefaultTrainer> trainer =
      std::make_unique<DefaultTrainer>(rand, inputs.begin(), inputs.end());
  std::cout << "Beginning training..." << std::endl;
  static volatile bool interrupted =
      false; // Static to allow use in the signal handler.
  signal(SIGINT, [](int) {
    interrupted = true;
    std::cout << "Stopping soon..." << std::endl;
  });
  while (!interrupted) {
    trainer->train(16, 0.01);
    std::cout << trainer->best().score << std::endl;
  }
  for (size_t i = 0; i < 10; i++) {
    const Input &input = inputs[i];
    const NetworkInputs<1> &networkOutputs =
        trainer->best().network.apply(input.networkInputs);
    std::cout << (input.correctlySpelled ? "true" : "false") << " "
              << networkOutputs[0] << std::endl;
  }
  return 0;
}
