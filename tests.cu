#include <stdio.h>
#include <assert.h>

#include "gpulife.hpp"

void test_modular_offset(){
  // A 2x3 grid:
  //    _____
  //   |0|1|2|
  //   |3|4|5|
  assert(MODULAR_OFFSET(0, 0, 2, 3) == 0);
  assert(MODULAR_OFFSET(0, 1, 2, 3) == 1);
  assert(MODULAR_OFFSET(0, 2, 2, 3) == 2);
  assert(MODULAR_OFFSET(1, 0, 2, 3) == 3);
  assert(MODULAR_OFFSET(1, 1, 2, 3) == 4);
  assert(MODULAR_OFFSET(1, 2, 2, 3) == 5);
}

int main(int argc, char** argv){
  assert(1);
  test_modular_offset();
  printf("w00t\n");
  return 0;
}

