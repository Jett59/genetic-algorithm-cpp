#ifndef GENETIC_RING_BUFFER_H
#define GENETIC_RING_BUFFER_H

#include "cpu.h"
#include <array>
#include <atomic>
#include <optional>

namespace genetic {
template <typename T, size_t length> class RingBuffer {
  struct Entry {
    std::optional<T> value;
    std::atomic_bool beingModified = false;

    Entry() = default;
    constexpr Entry(std::nullopt_t) : value(std::nullopt) {}
    constexpr Entry(const Entry &other) : value(other.value) {
      beingModified.store(other.beingModified.load());
    }
    constexpr Entry(Entry &&other) : value(std::move(other.value)) {
      beingModified.store(other.beingModified.load());
    }
    constexpr Entry &operator=(const Entry &other) {
      if (other.value) {
        value.emplace(*other.value);
      } else {
        value.reset();
      }
      beingModified.store(other.beingModified.load());
      return *this;
    }
    constexpr Entry &operator=(Entry &&other) {
      if (other.value) {
        value.emplace(std::move(*other.value));
      } else {
        value.reset();
      }
      beingModified.store(other.beingModified.load());
      return *this;
    }
  };

  std::atomic_size_t readIndex = 0;
  std::atomic_size_t writeIndex = 0;
  std::array<Entry, length> buffer;

public:
  RingBuffer() { std::fill(buffer.begin(), buffer.end(), Entry{std::nullopt}); }

  std::optional<T> tryRead() {
    while (true) {
      size_t readIndex = this->readIndex;
      if (readIndex == writeIndex) {
        return std::nullopt;
      }
      size_t newReadIndex = readIndex + 1 < length ? readIndex + 1 : 0;
      if (this->readIndex.compare_exchange_weak(readIndex, newReadIndex)) {
        while (true) {
          if (!buffer[readIndex].beingModified) {
            Entry result = buffer[readIndex];
            buffer[readIndex].value.reset();
            return result.value.value();
          } else {
            relaxCpu();
          }
        }
      }
    }
  }

  T read() {
    while (true) {
      auto result = tryRead();
      if (result) {
        return result.value();
      } else {
        relaxCpu();
      }
    }
  }

  void write(T value) {
    while (true) {
      size_t writeIndex = this->writeIndex;
      size_t newWriteIndex = writeIndex + 1 < length ? writeIndex + 1 : 0;
      if (newWriteIndex == readIndex) {
        relaxCpu();
        continue;
      }
      this->buffer[writeIndex].beingModified = true;
      if (this->writeIndex.compare_exchange_weak(writeIndex, newWriteIndex)) {
        this->buffer[writeIndex].value.emplace(std::forward<T>(value));
        this->buffer[writeIndex].beingModified = false;
        return;
      }
    }
  }
};
} // namespace genetic

#endif