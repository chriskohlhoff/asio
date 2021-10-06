#include "asio/experimental/compute/cuda/bulk.hpp"
#include <asio/io_context.hpp>
#include <cassert>
#include <algorithm>
#include <vector>
#include "asio/experimental/compute/cuda/command_queue.hpp"
#include "asio/experimental/compute/cuda/copy.hpp"
#include "asio/experimental/compute/cuda/device_vector.hpp"
#include "inline_executor.hpp"

namespace cuda = asio::experimental::compute::cuda;

void __device__ inner_print(std::size_t x)
{
  cudaError_t e;
  cuda::basic_command_queue<inline_executor, cudaError_t> child_q({}, e);
  bulk(child_q, 1,
      [x] __device__ (std::size_t)
      {
        printf("%llu\n", x);
      },
      [](cudaError_t e)
      {
      });
}

int main()
{
  try
  {
    asio::io_context io_ctx;
    cuda::command_queue command_queue(io_ctx.get_executor());

    constexpr std::size_t elems = 1'000'000;
    constexpr std::size_t batches = 100;
    constexpr std::size_t batch_size = elems / batches;

    std::vector<int> host_mem_1(elems);
    std::vector<int> host_mem_2(elems);
    cuda::device_vector<int> device_mem(elems);

    std::fill(host_mem_1.begin(), host_mem_1.end(), 42);

    bool called = false;
    copy(command_queue, host_mem_1.begin(), host_mem_1.end(), device_mem.begin(),
        [&](std::error_code e)
        {
          assert(!e);
          bulk(command_queue, batches,
              [mem = device_mem.data()] __device__ (std::size_t x)
              {
                std::size_t start = x * batch_size;
                std::size_t end = start + batch_size;
                for (std::size_t i = start; i < end; ++i)
                  mem[i] *= 2;
                inner_print(x);
              },
              [&](std::error_code e)
              {
                copy(command_queue, device_mem.begin(), device_mem.end(), host_mem_2.begin(),
                    [&](std::error_code e)
                    {
                      assert(!e);
                      called = true;
                    });
              });
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
