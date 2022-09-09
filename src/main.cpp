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

int main(int argc, char **argv) {
  DefaultRandom rand;
  std::vector<Input> inputs = readInputs("inputs.txt");
  std::cout << "Initializing the trainer..." << std::endl;
  std::unique_ptr<DefaultTrainer> trainer =
      std::make_unique<DefaultTrainer>(rand, inputs.begin(), inputs.end());
      if (argc > 1) {
        DefaultNetwork network;
        std::ifstream input(argv[1], std::ios::binary);
        input >> network;
        input.close();
        trainer->insertNetwork(network);
      }
  std::cout << "Beginning training..." << std::endl;
  static volatile bool interrupted =
      false; // Static to allow use in the signal handler.
  signal(SIGINT, [](int) {
    interrupted = true;
    std::cout << "Stopping soon..." << std::endl;
  });
  while (!interrupted) {
    trainer->train(16, 0.0001);
    std::cout << trainer->best().score << std::endl;
  }
  double accuracy = 0;
  for (const auto &input : inputs) {
    const NetworkInputs<1> &networkOutputs =
        trainer->best().network.apply(input.networkInputs);
    double networkOutput = networkOutputs[0];
    if (networkOutput >= 0.5 == input.correctlySpelled) {
      accuracy++;
    }
  }
  accuracy /= inputs.size();
  std::cout << "Accuracy: " << accuracy << std::endl;
  // Start an interractive session.
  std::cout << "Your turn:" << std::endl;
  while (true) {
    std::string word;
    std::cin >> word;
    if (word == "quit") {
      break;
    }
    Input input(word, false);
    NetworkInputs<1> networkOutputs =
        trainer->best().network.apply(input.networkInputs);
    double networkOutput = networkOutputs[0];
    std::cout << (networkOutput > 0.5 ? "true" : "false") << std::endl;
  }
  std::ofstream output("out.gnt", std::ios::binary);
  output << trainer->best().network;
  return 0;
}
