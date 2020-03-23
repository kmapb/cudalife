#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h>

// Square stencil.
template<typename Payload, typename Op, int kBlockHeight, int kBlockWidth>
__global__
void apply2dStencil(const Payload* in, Payload* out, int numRows, int numCols) {
  // XXX: build a shared "skirt" around the block.

  // const int blockUpperLeft = kBlockHeight * gridDim.x * blockIdx.y + blockIdx.x * kBlockWidth;
  // printf("(%d, %d) -> %d\n", blockIdx.x, blockIdx.y, blockUpperLeft);
  const int center = threadIdx.y * blockDim.x * gridDim.x + threadIdx.x;
  // printf("b(%d, %d) t(%d, %d)-> %d\n", blockIdx.x, blockIdx.y,
  //        threadIdx.x, threadIdx.y, center);

  // XXX: it's lame that we're screwing up the border.
  const int north = threadIdx.y >= 1 ? (threadIdx.y - 1) * blockDim.x +
    threadIdx.x : center;
  const int south = threadIdx.y < numCols - 1 ? (threadIdx.y + 1) * blockDim.x +
    threadIdx.x : center;
  Op op;
  out[center] = op.op(in[north - 1],  in[north], out[north + 1],
                      in[center - 1], in[center], out[center + 1],
                      in[south - 1],  in[south], out[south + 1]);
}

typedef uint8_t Cell;
struct GameOfLifeOp {
  __device__ Cell op(Cell nw, Cell n, Cell ne,
                     Cell w, Cell center, Cell e,
                     Cell sw, Cell s, Cell se) {
    int sum = nw + n + ne + w + e + sw + s +e;
    bool isThree = sum == 3;
    bool isTwo = sum == 2;
    bool isAlive = center;
    return uint8_t( (!isAlive & isThree) | (isAlive & (isThree | isTwo)));
  }
};

class GPULife {
  int m_numRows, m_numCols;
  Cell* m_gpuCells, *m_gpuCellsOut;
  Cell* m_hCells;

 public:
  GPULife(int numRows, int numCols) 
    : m_numRows(numRows)
    , m_numCols(numCols)
  {
    cudaMalloc(&m_gpuCells, 2 * numRows * numCols * sizeof(Cell));
    m_gpuCellsOut = m_gpuCells + numRows * numCols * sizeof(Cell);

    m_hCells = (Cell*)malloc(sizeof(Cell) * numRows * numCols);

    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        m_hCells[i * numCols + j] = bool(random() % 3);
      }
    }

    cudaMemcpy(m_gpuCells, m_hCells, sizeof(Cell) * numRows * numCols,
               cudaMemcpyHostToDevice);
  }

  ~GPULife() {
    cudaFree(m_gpuCells);
    free(m_hCells);
  }

  static int ceilDiv(int a, int b) { return ceil(float(a) / b); }

  void gen() {
    static const int kBlockWidth = 16;
    static const int kBlockHeight = 16;
    const dim3 grid(ceilDiv(m_numCols, kBlockWidth),
                       ceilDiv(m_numRows, kBlockHeight));
    const dim3 blk(kBlockWidth, kBlockHeight);
    apply2dStencil<Cell, GameOfLifeOp, kBlockHeight, kBlockHeight><<<grid, blk>>>(m_gpuCells, m_gpuCellsOut, m_numRows, m_numCols);
    Cell* temp = m_gpuCells;
    m_gpuCells = m_gpuCellsOut;
    m_gpuCellsOut = temp;
    cudaMemcpy(m_hCells, m_gpuCells, sizeof(Cell) * m_numRows * m_numCols,
               cudaMemcpyDeviceToHost);
  }

  void show() const {
    for (int i = 0; i < m_numRows; i++) {
      for (int j = 0; j < m_numCols; j++) {
        bool val = bool(m_hCells[i * m_numCols + j]);
        printf("%c", val ? '#' : ' ');
      }
      printf("\n");
    }
  }
};

int main(int argc, char** argv) {
  printf("shreck\n");
  GPULife board(24, 80);

  for (int i = 0; i < 100; i++) {
    printf("-----------------------\n");
    board.gen();
    board.show();
  }
  return 0;
}

