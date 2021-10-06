#include "asio/experimental/compute/cuda/device_vector.hpp"
#include <cassert>

int main()
{
  asio::experimental::compute::cuda::device_vector<int> values(100);
  assert(values.data() != nullptr);
  assert(values.size() == 100);
  assert(values.begin() + 100 == values.end());
}
