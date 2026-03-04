/*
 * window_buffer.h
 * Circular sliding window buffer — Phase 03
 *
 * Stores the last WINDOW_SIZE IMU samples.
 * Once full, every new push shifts the window forward
 * by one sample (stride=1), giving the model a fresh
 * inference on every new reading without waiting for
 * a full new window (reduces latency to ~20ms).
 */

#ifndef WINDOW_BUFFER_H
#define WINDOW_BUFFER_H

#define WINDOW_SIZE  50
#define N_FEATURES   6

class WindowBuffer {
public:
  WindowBuffer() : _count(0), _head(0) {
    memset(_buf, 0, sizeof(_buf));
  }

  // Push one 6-element sample into the circular buffer
  void push(float sample[N_FEATURES]) {
    for (int f = 0; f < N_FEATURES; f++) {
      _buf[_head][f] = sample[f];
    }
    _head = (_head + 1) % WINDOW_SIZE;
    if (_count < WINDOW_SIZE) _count++;
  }

  // True once the buffer has been filled at least once
  bool isFull() const {
    return _count >= WINDOW_SIZE;
  }

  // Copy buffer contents in chronological order into dst
  // dst must point to float[WINDOW_SIZE * N_FEATURES]
  void copyTo(float* dst) const {
    int start = _head; // oldest sample
    for (int t = 0; t < WINDOW_SIZE; t++) {
      int idx = (start + t) % WINDOW_SIZE;
      for (int f = 0; f < N_FEATURES; f++) {
        dst[t * N_FEATURES + f] = _buf[idx][f];
      }
    }
  }

  void reset() {
    _count = 0;
    _head  = 0;
    memset(_buf, 0, sizeof(_buf));
  }

private:
  float    _buf[WINDOW_SIZE][N_FEATURES];
  int      _count;
  int      _head;
};

#endif // WINDOW_BUFFER_H
