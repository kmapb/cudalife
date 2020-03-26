#include <stdio.h>
#include <string>
#include <vector>

#include "gpulife.hpp"

std::vector<std::string> errors;

#define asrt(x) do {   \
  auto pred = bool(x); \
  if (!pred) {         \
    errors.push_back(std::string(#x)); \
  }                    \
} while(0)

void test_modular_offset(){
  // A 2x3 grid:
  //    _____
  //   |0|1|2|
  //   |3|4|5|
  asrt(MODULAR_OFFSET(0, 0, 2, 3) == 0);
  asrt(MODULAR_OFFSET(0, 1, 2, 3) == 1);
  asrt(MODULAR_OFFSET(0, 2, 2, 3) == 2);
  asrt(MODULAR_OFFSET(1, 0, 2, 3) == 3);
  asrt(MODULAR_OFFSET(1, 1, 2, 3) == 4);
  asrt(MODULAR_OFFSET(1, 2, 2, 3) == 5);

  // The "apron": negative numbers
  asrt(MODULAR_OFFSET(-1, -1, 2, 3) == 5);

}

int main(int argc, char** argv){
  asrt(1);
  test_modular_offset();
  if (errors.size() == 0){
    printf("w00t\n");
    return 0;
  }

  for (auto e: errors) {
    fprintf(stderr, "Error: %s\n", e.c_str());
  }
}

