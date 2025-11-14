CXX := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic
LDFLAGS :=

TARGET := lbm_sim
SOURCES := main.cpp Inamuro.cpp InamuroSolver.cpp common.cpp
OBJECTS := $(SOURCES:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

