#pragma once

__device__
static inline unsigned computeModularOffset(int x, int y, size_t numRows, size_t numCols) {
  return MODULAR_OFFSET(x, y, numRows, numCols);
}

__device__
static inline Offset2D computeOffset(dim3 tileDim, dim3 blockIdx, uint3 threadIdx, size_t numRows, size_t numCols) {
  auto x = tileDim.x * blockIdx.x + threadIdx.x;
  auto y = tileDim.y * blockIdx.y + threadIdx.y;
  return { x, y };
}

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
  if (false) printf("----   %d %d -> %02d    ----\n"
         "nw(%02d) n(%02d) ne(%02d)\n"
         " w(%02d)   %02d   e(%02d)\n"
         "sw(%02d) s(%02d) se(%02d)\n",
         off2.x, off2.y,
         center,
         nudge(-1, -1), nudge(0, -1), nudge(+1, -1),
         nudge(-1, 0),  center,       nudge(+1, 0),
         nudge(-1, +1), nudge(0, +1), nudge(+1, +1));

  out[center] = Op::op(
        in[nudge(-1, -1)], in[nudge(0, -1)], in[nudge(+1, -1)],  // nw, north, ne
        in[nudge(-1,  0)], in[center],       in[nudge(+1,  0)],  // w        , e
        in[nudge(-1, +1)], in[nudge(0, +1)], in[nudge(+1, +1)]); // sw, south, se
}


