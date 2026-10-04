CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra
.PHONY: all check output-check test paper clean
all: code/separation.so code/law96.so
code/%.so: code/%.cpp
	$(CXX) $(CXXFLAGS) -shared $< -o $@
check: all
	OPENBLAS_NUM_THREADS=1 python code/compact_replay.py
output-check: all
	OPENBLAS_NUM_THREADS=1 python code/verify_output.py
test: all
	OPENBLAS_NUM_THREADS=1 python code/test_arithmetic.py
paper:
	cd paper && bash build.sh
clean:
	rm -f code/separation.so code/law96.so
