#ifndef GENETIC_WORKERS_H
#define GENETIC_WORKERS_H

#include <array>
#include <atomic>
#include <optional>
#include <semaphore>
#include <thread>
#include <vector>

namespace genetic {
template <typename Input, typename Output>
void add(Output &sum, const Output &additional) {
  sum += additional;
}

template <typename Input, typename Context, typename Output,
          Output (*handler)(const Input &, const Context &),
          void (*collector)(Output &, const Output &)>
struct Worker {
  static constexpr size_t maxJobs = 8192;

  std::thread thread;
  std::binary_semaphore semaphore;
  std::array<std::optional<Input>, maxJobs> inputs;
  const Context *context;
  size_t inputCount = 0;
  Output output;
  std::atomic_flag done = ATOMIC_FLAG_INIT;

  Worker() : semaphore(0) {
    thread = std::thread([this] {
      while (true) {
        semaphore.acquire();
        for (size_t i = 0; i < inputCount; i++) {
          auto &input = inputs[i];
          collector(output, handler(*input, *context));
          input = std::nullopt;
        }
        inputCount = 0;
        done.test_and_set();
      }
    });
  }

  template <typename Iter> void operator()(Iter begin, Iter end) {
    done.clear();
    output = {};
    for (auto iter = begin; iter != end; iter++) {
      inputs[inputCount++].emplace(*iter);
    }
    semaphore.release();
  }

  Output await() {
    while (!done.test()) {
      std::this_thread::yield();
    }
    return output;
  }
};

template <typename Input, typename Context, typename Output,
          Output (*handler)(const Input &, const Context &),
          void (*collector)(Output &, const Output &) = add<Input, Output>>
struct WorkerPool {
  static constexpr size_t maxWorkers = 64;
  std::array<Worker<Input, Context, Output, handler, collector>, maxWorkers>
      workers;
  size_t nextWorker = 0;
  size_t numWorkers;

  WorkerPool() : numWorkers(std::thread::hardware_concurrency()) {}
  // Deleted copy constructor and assignment operator.
  WorkerPool(const WorkerPool &) = delete;
  WorkerPool &operator=(const WorkerPool &) = delete;
  // Move constructors and assignment operator.
  WorkerPool(WorkerPool &&) = default;
  WorkerPool &operator=(WorkerPool &&) = default;

  template <typename Iter>
  void operator()(Iter begin, Iter end, const Context &context) {
    size_t numJobs = std::distance<Iter>(begin, end);
    size_t jobsPerWorker = numJobs / numWorkers;
    size_t remainingJobs = numJobs % numWorkers;
    for (size_t i = 0; i < numWorkers; i++) {
      size_t jobs = jobsPerWorker;
      if (remainingJobs > 0) {
        jobs++;
        remainingJobs--;
      }
      workers[i].context = &context;
      workers[i](begin, begin + jobs);
      begin += jobs;
    }
  }

  Output await() {
    Output output{};
    for (size_t i = 0; i < numWorkers; i++) {
      collector(output, workers[i].await());
    }
    return output;
  }
};
} // namespace genetic

#endif