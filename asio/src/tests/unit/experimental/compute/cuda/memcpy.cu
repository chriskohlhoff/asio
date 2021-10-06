#include "asio/experimental/compute/cuda/memcpy.hpp"
#include "asio/io_context.hpp"
#include <cassert>
#include <algorithm>
#include <vector>
#include "asio/experimental/compute/cuda/command_queue.hpp"
#include "asio/experimental/compute/cuda/device_vector.hpp"

namespace cuda = asio::experimental::compute::cuda;

int main()
{
  try
  {
    asio::io_context io_ctx;
    cuda::command_queue command_queue(io_ctx.get_executor());

    std::vector<int> host_mem_1(1'000'000);
    std::vector<int> host_mem_2(1'000'000);
    cuda::device_vector<int> device_mem(1'000'000);

    std::fill(host_mem_1.begin(), host_mem_1.end(), 42);

    bool called = false;
    memcpy_host_to_device(command_queue,
        device_mem.data(), &host_mem_1[0], host_mem_1.size() * sizeof(int),
        [&](std::error_code e)
        {
          assert(!e);
          memcpy_device_to_host(command_queue,
              &host_mem_2[0], device_mem.data(), device_mem.size() * sizeof(int),
              [&](std::error_code e)
              {
                assert(!e);
                called = true;
              });
        });

    assert(!called);
    io_ctx.run();
    assert(called);

    assert(host_mem_1 == host_mem_2);
  }
  catch (const std::exception&)
  {
    assert(0);
  }
}
