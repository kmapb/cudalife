CXX=/usr/bin/g++
grunk: grunk.cu
	nvcc -g -O3 -arch compute_30 --compiler-bindir ${CXX} -o $@  $^

