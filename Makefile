CXXFLAGS=-std=c++17 -O3 -march=native -ffast-math -Wall -Wextra -pedantic -g

SRCS=$(wildcard src/*.cpp)

a.out: $(SRCS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@

clean:
	rm -f a.out
