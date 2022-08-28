#ifndef GENETIC_WORKERS_H
#define GENETIC_WORKERS_H

#include "ringBuffer.h"
#include <array>
#include <atomic>
#include <optional>
#include <semaphore>
#include <thread>
#include <vector>

namespace genetic {
auto add(auto a, auto b) { return a + b; }

template <typename Input, typename Output, Output (*handler)(Input &),
          Output (*collector)(Output, Output) = add, size_t max = 65536>
struct WorkerPool {
  RingBuffer<Input, max> jobs;
  std::atomic<Output> output;
  std::atomic_size_t pendingJobs = 0;

  std::vector<std::thread> threads;

  WorkerPool() {
    auto numThreads = std::thread::hardware_concurrency();
    for (size_t i = 0; i < numThreads; i++) {
      threads.push_back(std::thread(
          [](WorkerPool *pool) {
            Output ourTotalOutput{};
            size_t completedJobs = 0;
            while (true) {
              auto initialReadAttempt = pool->jobs.tryRead();
              if (!initialReadAttempt) {
                pool->output = collector(pool->output, ourTotalOutput);
                pool->pendingJobs -= completedJobs;
                ourTotalOutput = 0;
                completedJobs = 0;
              }
              auto job =
                  initialReadAttempt ? *initialReadAttempt : pool->jobs.read();
              ourTotalOutput = collector(ourTotalOutput, handler(job));
              completedJobs++;
            }
          },
          this));
    }
  }
  // Deleted copy constructor and assignment operator.
  WorkerPool(const WorkerPool &) = delete;
  WorkerPool &operator=(const WorkerPool &) = delete;
  // Move constructors and assignment operator.
  WorkerPool(WorkerPool &&) = default;
  WorkerPool &operator=(WorkerPool &&) = default;

  void addJob(const Input &input) {
    pendingJobs++;
    jobs.write(input);
  }

  Output waitForJobs() {
    output = Output{};
    while (pendingJobs) {
      std::this_thread::yield();
    }
    Output result = output;
    output = Output{};
    return result;
  }
};
} // namespace genetic

#endif