CXXFLAGS=-O3 -march=native -ffast-math

SRCS=$(wildcard src/*.cpp)

a.out: $(SRCS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@

clean:
	rm -f a.out
