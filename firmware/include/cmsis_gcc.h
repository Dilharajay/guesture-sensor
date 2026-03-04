#ifndef __CMSIS_GCC_H
#define __CMSIS_GCC_H

#include <stdint.h>

#ifndef __ASM
#define __ASM __asm
#endif
#ifndef __INLINE
#define __INLINE inline
#endif
#ifndef __STATIC_INLINE
#define __STATIC_INLINE static inline
#endif
#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE __attribute__((always_inline)) static inline
#endif
#ifndef __NO_RETURN
#define __NO_RETURN __attribute__((noreturn))
#endif
#ifndef __USED
#define __USED __attribute__((used))
#endif
#ifndef __WEAK
#define __WEAK __attribute__((weak))
#endif
#ifndef __PACKED
#define __PACKED __attribute__((packed))
#endif
#ifndef __PACKED_STRUCT
#define __PACKED_STRUCT struct __attribute__((packed))
#endif
#ifndef __PACKED_UNION
#define __PACKED_UNION union __attribute__((packed))
#endif
#ifndef __ALIGNED
#define __ALIGNED(x) __attribute__((aligned(x)))
#endif
#ifndef __RESTRICT
#define __RESTRICT __restrict
#endif
#ifndef __COMPILER_BARRIER
#define __COMPILER_BARRIER() __ASM volatile("" ::: "memory")
#endif

__STATIC_FORCEINLINE uint8_t __CLZ(uint32_t data) {
  if (data == 0U) return 32U;
  uint32_t count = 0U;
  uint32_t mask = 0x80000000U;
  while ((data & mask) == 0U) {
    count += 1U;
    mask >>= 1U;
  }
  return (uint8_t)count;
}

__STATIC_FORCEINLINE int32_t __SSAT(int32_t val, uint32_t sat) {
  if ((sat >= 1U) && (sat <= 32U)) {
    const int32_t max = (int32_t)((1U << (sat - 1U)) - 1U);
    const int32_t min = -1 - max;
    if (val > max) return max;
    if (val < min) return min;
  }
  return val;
}

__STATIC_FORCEINLINE uint32_t __USAT(int32_t val, uint32_t sat) {
  if (sat <= 31U) {
    const uint32_t max = (1U << sat) - 1U;
    if (val > (int32_t)max) return max;
    if (val < 0) return 0U;
  }
  return (uint32_t)val;
}

__STATIC_FORCEINLINE uint32_t __ROR(uint32_t op1, uint32_t op2) {
  op2 %= 32U;
  if (op2 == 0U) return op1;
  return (op1 >> op2) | (op1 << (32U - op2));
}

#endif
