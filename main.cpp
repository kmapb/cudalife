#include <stdio.h>

#include "gpulife.hpp"

int main(int argc, char** argv) {
  printf("shreck\n");
  GPULife board(5, 8);

  for (int i = 0; i < 3; i++) {
    printf("-----------------------\n");
    board.gen();
    board.show();
  }
  return 0;
}

