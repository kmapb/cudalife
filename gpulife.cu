#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h>

#include "gpulife.hpp"

typedef struct Offset2D {
  unsigned x, y;
} Offset2D;

__device__
unsigned computeModularOffset(int x, int y, size_t numRows, size_t numCols) {
  return MODULAR_OFFSET(x, y, numRows, numCols);
}

__device__
Offset2D computeOffset(dim3 tileDim, dim3 blockIdx, uint3 threadIdx, size_t numRows, size_t numCols) {
  auto x = tileDim.x * blockIdx.x + threadIdx.x;
  auto y = tileDim.y * blockIdx.y + threadIdx.y;
  return { x, y };
}

static int ceilDiv(int a, int b) { return ceil(float(a) / b); }

// Square stencil.
template<typename Payload, typename Op, int kBlockHeight, int kBlockWidth>
__global__
void apply2dStencil(const Payload* in, Payload* out, dim3 tileDim, int numRows, int numCols) {
  auto off2 = computeOffset(tileDim, blockIdx, threadIdx, numRows, numCols);
  if (off2.x >= numCols) return;
  if (off2.y >= numRows) return;

  auto center = computeModularOffset(off2.x, off2.y, numRows, numCols);
  if (false) printf("b(%d,%d) t(%d,%d) / td (%d,%d) -> off2(%d,%d) mod(%d)\n",
        blockIdx.x, blockIdx.y,
        threadIdx.x, threadIdx.y,
        tileDim.x, tileDim.y,
        off2.x, off2.y,
        center);

  auto nudge = [&](int dx, int dy) {
    return computeModularOffset(off2.x + dx, off2.y + dy, numRows, numCols);
  };
  Op op;
  if (false) printf("----   %d %d -> %02d    ----\n"
         "nw(%02d) n(%02d) ne(%02d)\n"
         " w(%02d)   %02d   e(%02d)\n"
         "sw(%02d) s(%02d) se(%02d)\n",
         off2.x, off2.y,
         center,
         nudge(-1, -1), nudge(0, -1), nudge(+1, -1),
         nudge(-1, 0),  center,       nudge(0, +1),
         nudge(-1, +1), nudge(0, +1), nudge(+1, +1));

  out[center] = op.op(
        in[nudge(-1, -1)], in[nudge(0, -1)], in[nudge(+1, -1)],  // nw, north, ne
        in[nudge(-1,  0)], in[center],       in[nudge( 0, +1)],  // w        , e
        in[nudge(-1, +1)], in[nudge(0, +1)], in[nudge(+1, +1)]); // sw, south, se
}

struct GameOfLifeOp {
  __device__ Cell op(Cell nw, Cell n, Cell ne,
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

  m_hCells = (Cell*)malloc(sizeof(Cell) * numRows * numCols);

  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++) {
      m_hCells[i * numCols + j] = bool(random() % 3);
    }
  }

  cudaMemcpy(m_gpuCells, m_hCells, sizeof(Cell) * numRows * numCols,
             cudaMemcpyHostToDevice);
}

GPULife::~GPULife() {
    cudaFree(m_gpuCells);
    free(m_hCells);
}

void GPULife::show() const {
  cudaMemcpy(m_hCells, m_gpuCells, sizeof(Cell) * m_numRows * m_numCols,
             cudaMemcpyDeviceToHost);
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
  static const int kBlockWidth = 16;
  static const int kBlockHeight = 16;
  const dim3 grid(ceilDiv(m_numCols, kBlockHeight),
                  ceilDiv(m_numRows, kBlockWidth));
  const dim3 blk(kBlockWidth, kBlockHeight);
  const dim3 tile(kBlockWidth, kBlockHeight);
  apply2dStencil<Cell, GameOfLifeOp, kBlockWidth, kBlockHeight><<<grid, blk>>>(m_gpuCells, m_gpuCellsOut, tile, m_numRows, m_numCols);
  // Flip the buffer
  Cell* temp = m_gpuCells;
  m_gpuCells = m_gpuCellsOut;
  m_gpuCellsOut = temp;
}
