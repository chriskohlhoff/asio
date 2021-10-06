#include "asio/experimental/compute/cuda/error.hpp"
#include <cassert>

int main()
{
  std::error_code e1 = cudaErrorInvalidValue;
  assert(e1 == cudaErrorInvalidValue);
  assert(e1.message() == "invalid argument");
}
