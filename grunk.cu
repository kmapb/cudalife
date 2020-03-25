#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h>

typedef struct Offset2D {
  size_t x, y;
} Offset2ds;

__device__
size_t computeModularOffset(size_t x, size_t y, size_t numRows, size_t numCols) {
  return (x % numCols) * numCols + (y % numRows);
}

__device__
Offset2D computeOffset(dim3 tileDim, dim3 blockIdx, uint3 threadIdx, size_t numRows, size_t numCols) {
  auto row = tileDim.y * blockIdx.y + threadIdx.y;
  auto col = tileDim.x * blockIdx.x + threadIdx.x;
  return Offset2D { col, row };
}

// Square stencil.
template<typename Payload, typename Op, int kBlockHeight, int kBlockWidth>
__global__
void apply2dStencil(const Payload* in, Payload* out, dim3 tileDim, int numRows, int numCols) {
  auto off2 = computeOffset(tileDim, blockIdx, threadIdx, numRows, numCols);
  if (true || blockIdx.x == 3 && blockIdx.y == 1){
    // printf("gridDim: %d,%d\n", gridDim.x, gridDim.y);
    printf("b(%d, %d) t(%d, %d)-> %d,%d\n", blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y, int(off2.x), int(off2.y));
  }

  Op op;
  if (off2.x >= numCols) return;
  if (off2.y >= numRows) return;

  auto center = computeModularOffset(off2.x, off2.y, numRows, numCols);
  if (false && off2.x >= 1 && off2.x < numCols - 1 &&
      off2.y >= 1 && off2.y < numRows - 1) {
    auto north = center - numCols;
    auto south = center + numCols;
    out[center] = op.op(in[north - 1],  in[north], out[north + 1],
                        in[center - 1], in[center], out[center + 1],
                        in[south - 1],  in[south], out[south + 1]);
    return;
  }

  auto nudge = [&](int dx, int dy) {
    return computeModularOffset(off2.x + dx, off2.y + dy, numRows, numCols);
  };
  out[center] = op.op(
        in[nudge(-1, -1)], in[nudge(0, -1)], in[nudge(+1, -1)],  // nw, north, ne
        in[nudge(-1,  0)], in[center],       in[nudge( 0, +1)],  // w        , e
        in[nudge(-1, +1)], in[nudge(0, +1)], in[nudge(+1, +1)]); // sw, south, se
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

  ~GPULife() {
    cudaFree(m_gpuCells);
    free(m_hCells);
  }

  static int ceilDiv(int a, int b) { return ceil(float(a) / b); }

  void gen() {
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

  void show() const {
    cudaMemcpy(m_hCells, m_gpuCells, sizeof(Cell) * m_numRows * m_numCols,
               cudaMemcpyDeviceToHost);
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

