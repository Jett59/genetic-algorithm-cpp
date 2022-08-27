#ifndef GENETIC_CPU_H
#define GENETIC_CPU_H

namespace genetic {
static inline void relaxCpu() {
#if defined(__x86_64__)
  __asm__ __volatile__("pause");
#elif defined(__arm__)
  __asm__ __volatile__("yield");
#else
#warning No yield / pause instruction
#endif
}
} // namespace genetic

#endif