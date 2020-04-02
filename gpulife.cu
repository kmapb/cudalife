#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h>

#include "gpulife.hpp"
#include "2dstencil.cuh"

struct GameOfLifeOp {
  static __device__ Cell op(Cell nw, Cell n, Cell ne,
                     Cell w, Cell center, Cell e,
                     Cell sw, Cell s, Cell se) {
    int sum = nw + n + ne + w + e + sw + s + se;
    uint8_t isThree = sum == 3;
    uint8_t isTwo = sum == 2;
    uint8_t isAlive = center;
    return uint8_t( (~isAlive & isThree) | (isAlive & (isThree | isTwo)));
  }
};

// GPULife methods
GPULife::GPULife(int numRows, int numCols) 
  : m_numRows(numRows)
  , m_numCols(numCols)
{
  const auto boardSize = numRows * numCols * sizeof(Cell);
  cudaMalloc(&m_gpuCells, 2 * boardSize);
  m_gpuCellsOut = m_gpuCells + boardSize;

  m_hCells = (Cell*)malloc(boardSize);
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++) {
      m_hCells[i * numCols + j] = bool(random() % 3);
    }
  }

  cudaMemcpy(m_gpuCells, m_hCells, sizeof(Cell) * m_numRows * m_numCols,
             cudaMemcpyHostToDevice);
}

void GPULife::sync() const {
  cudaMemcpy(m_hCells, m_gpuCells, sizeof(Cell) * m_numRows * m_numCols,
             cudaMemcpyDeviceToHost);
}

GPULife::~GPULife() {
    cudaFree(m_gpuCells);
    free(m_hCells);
}

void GPULife::show() const {
  sync();
  printf( "%c[2J", 27 );
  // Headers across top; first, column 10s
  printf("%14s", " ");
  for (int i = 10; i < m_numCols; i++){
    int head = i / 10;
    if (i % 10 == 0) printf("%d", head); else printf(" ");
  }
  printf("\n");

  // column 1's
  printf("    ");
  for (int i = 0; i < m_numCols; i++){
    int head = i % 10;
    printf("%d", head);
  }
  printf("\n");

  for (int i = 0; i < m_numRows; i++) {
    printf("%-4d", i);
    for (int j = 0; j < m_numCols; j++) {
      bool val = bool(m_hCells[i * m_numCols + j]);
      printf("%c", val ? '#' : ' ');
    }
    printf("\n");
  }
}

void GPULife::gen() {
  static const int kBlockWidth = 8;
  static const int kBlockHeight = 64;
  const dim3 grid(ceilDiv(m_numCols, kBlockHeight),
                  ceilDiv(m_numRows, kBlockWidth));
  const dim3 blk(kBlockWidth, kBlockHeight);
  const dim3 tile(kBlockWidth, kBlockHeight);
  apply2dStencil<Cell, GameOfLifeOp, kBlockWidth, kBlockHeight><<<grid, blk>>>(m_gpuCells, m_gpuCellsOut, tile, m_numRows, m_numCols);

  // Flip the buffer
  Cell* temp = m_gpuCells;
  m_gpuCells = m_gpuCellsOut;
  m_gpuCellsOut = temp;
  sync();
}

Offset2D GPULife::dims() const {
    return { unsigned(m_numCols), unsigned(m_numRows) };
}
