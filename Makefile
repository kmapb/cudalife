CXX=/usr/bin/g++
CXXFLAGS=-g -O3

grunk:	grunk.o main.o
	nvcc -g -o $@ $^

clean:
	rm -f *.o

tests: grunk.o tests.o
	nvcc -g -o $@ $^

grunk.o: grunk.cu gpulife.hpp
	nvcc $(CXXFLAGS) -arch compute_30 --compiler-bindir ${CXX} -c  grunk.cu

tests.o: tests.cu gpulife.hpp
	nvcc $(CXXFLAGS) -arch compute_30 --compiler-bindir ${CXX} -c  tests.cu

main.o: main.cpp gpulife.hpp
	$(CXX) -c $(CXXFLAGS) main.cpp

