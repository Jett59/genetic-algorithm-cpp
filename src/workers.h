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
template <typename Input, typename Output>
void add(std::atomic<Output> &sum, Output additional) {
  sum += additional;
}

template <typename Input, typename Output, Output (*handler)(Input &),
          void (*collector)(std::atomic<Output> &, Output) = add<Input, Output>,
          size_t max = 65536>
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
            std::atomic<Output> ourTotalOutput;
            size_t completedJobs = 0;
            while (true) {
              auto initialReadAttempt = pool->jobs.tryRead();
              if (!initialReadAttempt) {
                collector(pool->output, ourTotalOutput);
                pool->pendingJobs -= completedJobs;
                ourTotalOutput = 0;
                completedJobs = 0;
              }
              auto job =
                  initialReadAttempt ? *initialReadAttempt : pool->jobs.read();
              collector(ourTotalOutput, handler(job));
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
    jobs.write(input);
    pendingJobs++;
  }

  Output waitForJobs() {
    while (pendingJobs) {
      std::this_thread::yield();
    }
    Output result = output;
    output = {};
    return result;
  }
};
} // namespace genetic

#endif