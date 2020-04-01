#include <stdio.h>
#include <string>
#include <vector>
#include <stdint.h>
#include <assert.h>

#include "gpulife.hpp"
#include "2dstencil.cuh"

struct Failure {
  std::string message;
  std::string file;
  int line;
};

std::vector<Failure> g_errors;
int g_successes;

#define asrt(x) do {   \
  auto pred = bool(x); \
  if (pred) g_successes++; \
  if (!pred) {         \
    g_errors.push_back({ std::string(#x), std::string(__FILE__), __LINE__ }); \
  }                    \
} while(0)

static inline size_t modularOffset(int x, int y, size_t numRows, size_t numCols) {
  return MODULAR_OFFSET(x, y, numRows, numCols);

}

void test_modular_offset(){
  // A 2x3 grid:
  //    _____
  //   |0|1|2|
  //   |3|4|5|
  asrt(modularOffset(0, 0, 2, 3) == 0);
  asrt(modularOffset(1, 0, 2, 3) == 1);
  asrt(modularOffset(2, 0, 2, 3) == 2);
  asrt(modularOffset(0, 1, 2, 3) == 3);
  asrt(modularOffset(1, 1, 2, 3) == 4);
  asrt(modularOffset(2, 1, 2, 3) == 5);

  // The "apron": negative numbers
  asrt(modularOffset(-1u, -1u, 2, 3) == 5);

}

struct TestStencilOp {
  protected:
  unsigned x_, y_;
  uint64_t* cudaIn_;
  uint64_t* cudaOut_;


  size_t numBytes() const {
    return sizeof(uint64_t) * x_ * y_;
  }

  public:

  TestStencilOp(unsigned x, unsigned y): x_(x), y_(y) {
    auto s = cudaMalloc(&cudaIn_, numBytes());
    assert(s == cudaSuccess);
    s = cudaMalloc(&cudaOut_, numBytes());
    assert(s == cudaSuccess);
  }

  ~TestStencilOp() {
    cudaFree(cudaIn_);
    cudaFree(cudaOut_);
  }

  static __device__ uint64_t op(
        uint64_t nw, uint64_t n, uint64_t ne,
        uint64_t w,  uint64_t c, uint64_t e,
        uint64_t sw, uint64_t s, uint64_t se){
    return 711u;
  }
 
  void reset() {
    cudaMemset(cudaOut_, 0, numBytes());
  }

  void launch() {
    static const int kBlockWidth = 11;
    static const int kBlockHeight = 73;
    const dim3 grid(ceilDiv(x_, kBlockWidth),
              ceilDiv(y_, kBlockHeight));
    const dim3 blk(kBlockWidth, kBlockHeight);
    const dim3 tile(kBlockWidth,kBlockHeight);
    apply2dStencil<uint64_t, TestStencilOp, kBlockWidth, kBlockHeight><<<grid, blk>>>(
        cudaIn_, cudaOut_, tile, y_, x_);
  }

  bool verify() const {
    std::vector<uint64_t> hostmem;
    hostmem.resize(x_ * y_);
    cudaMemcpy(&hostmem[0], cudaOut_, numBytes(), cudaMemcpyDeviceToHost);
    for (auto u: hostmem){
      asrt(u == 711u);
    }
    return true;
  }
};

void test_2d_stencils() {
  for (auto x: { 1, 3, 7, 19, 1024 }) {
    for (auto y: { 1, 3, 7, 19, 1024 }) {
      TestStencilOp op(x, y);
      op.reset();
      op.launch();
      op.verify();
    }
  }
}

int main(int argc, char** argv){
  asrt(1);
  test_modular_offset();
  test_2d_stencils();
  if (g_errors.size() == 0){
    printf("Success: %d tests\n", g_successes);
    return 0;
  }

  for (const auto &e: g_errors) {
    fprintf(stderr, "Error: %s, %s, %d\n", e.message.c_str(), e.file.c_str(), e.line);
  }
  return 1;
}

