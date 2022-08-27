#ifndef GENETIC_RING_BUFFER_H
#define GENETIC_RING_BUFFER_H

#include "cpu.h"
#include "spinlock.h"
#include <array>
#include <atomic>
#include <optional>

namespace genetic {
template <typename T, size_t length> class RingBuffer {
  std::atomic_size_t readIndex = 0, writeIndex = 0, size = 0;
  std::array<std::optional<T>, length> buffer;
  Spinlock lock;

public:
  T read() {
    while (true) {
      auto result = tryRead();
      if (result) {
        return *result;
      } else {
        relaxCpu();
      }
    }
  }
  std::optional<T> tryRead() {
    lock.acquire();
    if (size == 0) [[unlikely]] {
      lock.release();
      return std::nullopt;
    }
    T result = *buffer[readIndex];
    buffer[readIndex++] = std::nullopt;
    if (readIndex >= length) {
      readIndex = 0;
    }
    size--;
    lock.release();
    return result;
  }
  void write(T value) {
    while (true) {
      while (size == length)
        [[unlikely]] { relaxCpu(); }
      lock.acquire();
      if (size == length) [[unlikely]] {
        lock.release();
        continue;
      }
      buffer[writeIndex++].emplace(value);
      if (writeIndex >= length) {
        writeIndex = 0;
      }
      size++;
      lock.release();
      return;
    }
  }
  bool empty() { return length == 0; }
};
} // namespace genetic

#endif