#include "asio/experimental/compute/copy.hpp"
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
    copy(command_queue, host_mem_1.begin(), host_mem_1.end(), device_mem.begin(),
        [&](std::error_code e)
        {
          assert(!e);
          copy(command_queue, device_mem.begin(), device_mem.end(), host_mem_2.begin(),
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
