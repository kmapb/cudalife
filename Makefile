CXX=/usr/bin/g++
CXXFLAGS=-g

default: xgpulife gpulife runtests

gpulife:	gpulife.o main.o
	nvcc -g -o $@ $^

xgpulife:	gpulife.o x11main.o
	nvcc -g -o $@ $^ -lX11

clean:
	rm -f *.o
	rm -f tests
	rm -f gpulife

runtests: tests
	./tests

tests: gpulife.o tests.o
	nvcc -g -o $@ $^

gpulife.o: gpulife.cu gpulife.hpp
	nvcc -O3 $(CXXFLAGS) -arch compute_30 --compiler-bindir ${CXX} -c  gpulife.cu

tests.o: tests.cu gpulife.hpp
	nvcc -g -arch compute_30 --compiler-bindir ${CXX} -c  tests.cu

main.o: main.cpp gpulife.hpp
	$(CXX) -c $(CXXFLAGS) main.cpp

