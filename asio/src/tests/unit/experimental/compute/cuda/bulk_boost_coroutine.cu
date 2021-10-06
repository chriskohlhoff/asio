#include "asio/experimental/compute/cuda/bulk.hpp"
#include "asio/experimental/compute/cuda/command_queue.hpp"
#include "asio/experimental/compute/cuda/copy.hpp"
#include "asio/experimental/compute/cuda/device_vector.hpp"
#include "asio/experimental/deferred.hpp"
#include "asio/experimental/linked_group.hpp"
#include "asio/io_context.hpp"
#include "asio/spawn.hpp"
#include <cassert>
#include <vector>
#include <iostream>

using asio::experimental::deferred;
using asio::any_io_executor;
using asio::yield_context;
namespace cuda = asio::experimental::compute::cuda;

// This is a coroutine
void demo(any_io_executor exec, yield_context yield)
{
  cuda::command_queue command_queue(exec);

  constexpr std::size_t elems = 10'000'000;
  constexpr std::size_t batches = 1'000;
  constexpr std::size_t batch_size = elems / batches;

  std::vector<int> host_mem_1(elems);
  std::vector<int> host_mem_2(elems);
  cuda::device_vector<int> device_mem(elems);

  std::fill(host_mem_1.begin(), host_mem_1.end(), 42);

  auto send = copy(command_queue,
      host_mem_1.begin(), host_mem_1.end(),
      device_mem.begin(), deferred);

  auto bulk_op = bulk(command_queue, batches,
      [mem = device_mem.data()] __device__ (std::size_t x)
      {
        std::size_t start = x * batch_size;
        std::size_t end = start + batch_size;
        for (std::size_t i = start; i < end; ++i)
          mem[i] *= 2;
      }, deferred);

  auto collect = copy(command_queue,
      device_mem.begin(), device_mem.end(),
      host_mem_2.begin(), deferred);

  // asynchronously run the linked group on the GPU and complete on the IO
  // executor.
  std::error_code ec;
  make_linked_group(
      send,
      bulk_op,
      collect
    ).async_wait(yield[ec]);

  if (ec)
    std::cout << "Error: " << ec.message() << '\n';
  else
    std::cout << "Success\n";

  assert(!ec);

  assert(host_mem_1.size() == host_mem_2.size());
  for (std::size_t i = 0; i < host_mem_1.size(); ++i)
    assert(host_mem_1[i] * 2 == host_mem_2[i]);
}

int main()
{
  asio::io_context io_ctx;
  spawn(io_ctx, [e = io_ctx.get_executor()](auto yield) { demo(e, yield); });
  io_ctx.run();
}
