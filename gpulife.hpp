#pragma once

#include <stdint.h>

typedef uint8_t Cell;

typedef struct Offset2D {
  unsigned x, y;
} Offset2D;

class GPULife {
  int m_numRows, m_numCols;
  Cell* m_gpuCells, *m_gpuCellsOut;
  Cell* m_hCells;

private:
  void sync() const;
 public:
  GPULife(int numRows, int numCols);
  ~GPULife();

  void gen();

  Offset2D dims() const; 
  const Cell* cells() const { sync(); return m_hCells; }
  void show() const;
};

// Macro so it works across everyting
#define MODULAR_OFFSET(x, y, numRows, numCols) \
    ((((x + numCols) % numCols)) + ((y + numRows) % numRows) * numCols)
