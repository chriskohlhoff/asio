#include "asio/experimental/compute/cuda/command_queue.hpp"
#include "asio/io_context.hpp"
#include <cassert>

namespace cuda = asio::experimental::compute::cuda;

int main()
{
  try
  {
    asio::io_context io_ctx;
    cuda::command_queue command_queue(io_ctx.get_executor());
    assert(command_queue.get_executor() == io_ctx.get_executor());
    assert(command_queue.native_handle() != nullptr);

    bool called = false;
    command_queue.async_submit(
        [](cuda::command_queue& cq)
        {
          return std::error_code{};
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
