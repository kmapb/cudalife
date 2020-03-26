#pragma once

#include <stdint.h>

typedef uint8_t Cell;

class GPULife {
  int m_numRows, m_numCols;
  Cell* m_gpuCells, *m_gpuCellsOut;
  Cell* m_hCells;

 public:
  GPULife(int numRows, int numCols);
  ~GPULife();


  void gen();
  void show() const;
};
