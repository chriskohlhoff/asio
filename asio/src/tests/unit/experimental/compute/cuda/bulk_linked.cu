#include "asio/experimental/compute/cuda/bulk.hpp"
#include <asio/experimental/deferred.hpp>
#include <asio/io_context.hpp>
#include <cassert>
#include <algorithm>
#include <vector>
#include "asio/experimental/compute/cuda/command_queue.hpp"
#include "asio/experimental/compute/cuda/copy.hpp"
#include "asio/experimental/compute/cuda/device_vector.hpp"
#include "asio/experimental/linked_group.hpp"

using asio::experimental::deferred;
namespace cuda = asio::experimental::compute::cuda;

int main()
{
  try
  {
    asio::io_context io_ctx;
    cuda::command_queue command_queue(io_ctx.get_executor());

    constexpr std::size_t elems = 10'000'000;
    constexpr std::size_t batches = 1'000;
    constexpr std::size_t batch_size = elems / batches;

    std::vector<int> host_mem_1(elems);
    std::vector<int> host_mem_2(elems);
    cuda::device_vector<int> device_mem(elems);

    std::fill(host_mem_1.begin(), host_mem_1.end(), 42);

    bool called = false;
    make_linked_group(
        copy(command_queue, host_mem_1.begin(), host_mem_1.end(), device_mem.begin(), deferred),
        bulk(command_queue, batches,
          [mem = device_mem.data()] __device__ (std::size_t x)
          {
            std::size_t start = x * batch_size;
            std::size_t end = start + batch_size;
            for (std::size_t i = start; i < end; ++i)
              mem[i] *= 2;
          }, deferred),
        copy(command_queue, device_mem.begin(), device_mem.end(), host_mem_2.begin(), deferred)
      ).async_wait(
        [&](std::error_code e)
        {
          assert(!e);
          called = true;
        });

    assert(!called);
    io_ctx.run();
    assert(called);

    assert(host_mem_1.size() == host_mem_2.size());
    for (std::size_t i = 0; i < host_mem_1.size(); ++i)
      assert(host_mem_1[i] * 2 == host_mem_2[i]);
  }
  catch (const std::exception&)
  {
    assert(0);
  }
}
