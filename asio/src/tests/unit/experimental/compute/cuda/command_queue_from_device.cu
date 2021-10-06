#include "asio/experimental/compute/cuda/command_queue.hpp"
#include "asio/io_context.hpp"
#include <cassert>
#include "inline_executor.hpp"

namespace cuda = asio::experimental::compute::cuda;

void __global__ child()
{
  printf("child\n");
}

void __device__ parent()
{
  cudaError_t e;
  cuda::basic_command_queue<inline_executor, cudaError_t> command_queue_2(inline_executor{}, e);
  command_queue_2.async_submit(
      [&] __device__(cuda::basic_command_queue<inline_executor, cudaError_t>& ctx)
      {
        child<<<1, 1, 0, ctx.native_handle()>>>();
        return cudaPeekAtLastError();
      },
      [] __device__(cudaError_t e)
      {
        printf("done\n");
      });
}

void __global__ parent_proxy()
{
  parent();
}

int main()
{
  try
  {
    asio::io_context io_ctx;

    cuda::command_queue command_queue_1(io_ctx.get_executor());
    assert(command_queue_1.get_executor() == io_ctx.get_executor());
    assert(command_queue_1.native_handle() != nullptr);

    bool called = false;
    command_queue_1.async_submit(
        [&](cuda::command_queue& ctx)
        {
          parent_proxy<<<1, 1, 0, ctx.native_handle()>>>();
          return cudaPeekAtLastError();
        },
        [&](std::error_code e)
        {
          assert(!e);
          called = true;
        });

    assert(!called);
    io_ctx.run();
    assert(called);
  }
  catch (const std::exception&)
  {
    assert(0);
  }
}
