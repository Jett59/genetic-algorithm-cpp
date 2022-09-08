CXXFLAGS=-std=c++20 -Ofast -march=native -ffast-math -Wall -Wextra -pedantic -g -pthread

SRCS=$(wildcard src/*.cpp)

a.out: $(SRCS) $(wildcard src/*.h) Makefile
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(SRCS) -o $@

clean:
	rm -f a.out
