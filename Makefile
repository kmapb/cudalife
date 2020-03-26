CXX=/usr/bin/g++
CXXFLAGS=-g -O3

grunk:	grunk.o main.o
	nvcc -g -o $@ $^

grunk.o: grunk.cu gpulife.hpp
	nvcc $(CXXFLAGS) -arch compute_30 --compiler-bindir ${CXX} -c  grunk.cu

main.o: main.cpp gpulife.hpp
	$(CXX) -c $(CXXFLAGS) main.cpp

