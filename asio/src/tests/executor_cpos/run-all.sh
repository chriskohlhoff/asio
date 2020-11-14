./run-all-gcc-tests.sh # warmup
./run-all-gcc-tests.sh | tee gcc-results.txt
./run-all-clang-tests.sh # warmup
./run-all-clang-tests.sh | tee clang-results.txt
