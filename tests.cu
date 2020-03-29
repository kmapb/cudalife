#include <stdio.h>
#include <string>
#include <vector>

#include "gpulife.hpp"

struct Failure {
  std::string message;
  std::string file;
  int line;
};

std::vector<Failure> errors;

#define asrt(x) do {   \
  auto pred = bool(x); \
  if (!pred) {         \
    errors.push_back({ std::string(#x), std::string(__FILE__), __LINE__ }); \
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

int main(int argc, char** argv){
  asrt(1);
  test_modular_offset();
  if (errors.size() == 0){
    printf("w00t\n");
    return 0;
  }

  for (const auto &e: errors) {
    fprintf(stderr, "Error: %s, %s, %d\n", e.message.c_str(), e.file.c_str(), e.line);
  }
}

