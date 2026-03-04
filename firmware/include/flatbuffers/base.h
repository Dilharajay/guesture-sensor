#pragma once

#include_next <flatbuffers/base.h>

#if defined(ESP8266)
#include <pgmspace.h>
#include <stdint.h>

namespace flatbuffers {

template<>
inline uint8_t ReadScalar<uint8_t>(const void *p) {
  return static_cast<uint8_t>(pgm_read_byte(p));
}

template<>
inline int8_t ReadScalar<int8_t>(const void *p) {
  return static_cast<int8_t>(pgm_read_byte(p));
}

template<>
inline uint16_t ReadScalar<uint16_t>(const void *p) {
  return EndianScalar(static_cast<uint16_t>(pgm_read_word(p)));
}

template<>
inline int16_t ReadScalar<int16_t>(const void *p) {
  return EndianScalar(static_cast<int16_t>(pgm_read_word(p)));
}

template<>
inline uint32_t ReadScalar<uint32_t>(const void *p) {
  return EndianScalar(static_cast<uint32_t>(pgm_read_dword(p)));
}

template<>
inline int32_t ReadScalar<int32_t>(const void *p) {
  return EndianScalar(static_cast<int32_t>(pgm_read_dword(p)));
}

template<>
inline uint64_t ReadScalar<uint64_t>(const void *p) {
  const uint8_t *b = reinterpret_cast<const uint8_t *>(p);
  const uint64_t lo = static_cast<uint64_t>(pgm_read_dword(b));
  const uint64_t hi = static_cast<uint64_t>(pgm_read_dword(b + 4));
  return EndianScalar((hi << 32) | lo);
}

template<>
inline int64_t ReadScalar<int64_t>(const void *p) {
  return static_cast<int64_t>(ReadScalar<uint64_t>(p));
}

template<>
inline float ReadScalar<float>(const void *p) {
  return EndianScalar(pgm_read_float(p));
}

template<>
inline double ReadScalar<double>(const void *p) {
  return EndianScalar(pgm_read_double(p));
}

}  // namespace flatbuffers
#endif
