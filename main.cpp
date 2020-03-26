#include <stdio.h>

#include "gpulife.hpp"

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

