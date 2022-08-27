#ifndef GENETIC_SPINLOCK_H
#define GENETIC_SPINLOCK_H

#include <atomic>
#include "cpu.h"

namespace genetic {
class Spinlock {
  std::atomic_flag lock = ATOMIC_FLAG_INIT;

public:
  Spinlock() { lock.clear(); }

  void acquire() {
    while (true) {
      while (lock.test()) {
        relaxCpu();
      }
      if (!lock.test_and_set()) {
        break;
      }
    }
  }
    void release() { lock.clear(); }
};
} // namespace genetic

#endif