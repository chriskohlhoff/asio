//
// checsksum_server_linked.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio.hpp>
#include <string>
#include <asio/experimental/compute/bulk.hpp>
#include <asio/experimental/compute/copy.hpp>
#include <asio/experimental/compute/cuda/command_queue.hpp>
#include <asio/experimental/compute/cuda/device_vector.hpp>
#include <asio/experimental/spawn.hpp>

using asio::buffer;
using asio::detached;
using asio::ip::tcp;
using asio::yield_context;
using asio::experimental::spawn;
namespace compute = asio::experimental::compute;
namespace cuda = asio::experimental::compute::cuda;

struct checksum_chunks_impl
{
  std::size_t num_chunks;
  std::size_t chunk_size;
  const unsigned char* data;
  unsigned int* results;

  __device__ void operator()(std::size_t x)
  {
    std::size_t start = x * chunk_size;
    std::size_t end = start + chunk_size;
    results[x] = 0;
    for (std::size_t i = start; i < end; ++i)
      results[x] = (results[x] + data[i]) % 256;
  }
};

template <class CompletionToken>
auto checksum_chunks(cuda::command_queue& cq, const cuda::device_vector<unsigned char>& data,
    cuda::device_vector<unsigned int>& results, CompletionToken&& token)
{
  checksum_chunks_impl impl =
  {
    .num_chunks = results.size(),
    .chunk_size = data.size() / results.size(),
    .data = data.data(),
    .results = results.data()
  };

  return bulk(cq, results.size(), impl, std::forward<CompletionToken>(token));
}

struct reduce_results_impl
{
  std::size_t num_chunks;
  unsigned int* results;

  __device__ void operator()(std::size_t x)
  {
    unsigned int result = 0;
    for (std::size_t i = 0; i < num_chunks; ++i)
      result = (result + results[i]) % 256;
    results[0] = result;
  }
};

template <class CompletionToken>
auto reduce_results(cuda::command_queue& cq,
    cuda::device_vector<unsigned int>& results, CompletionToken&& token)
{
  reduce_results_impl impl =
  {
    .num_chunks = results.size(),
    .results = results.data()
  };

  return bulk(cq, 1, impl, std::forward<CompletionToken>(token));
}

void checksum(tcp::socket s, yield_context yield)
{
  try
  {
    constexpr std::size_t chunk_size = 1024 * 1024;
    constexpr std::size_t num_chunks = 16;
    constexpr std::size_t message_size = num_chunks * chunk_size;

    cuda::command_queue command_queue(s.get_executor());

    std::vector<unsigned char> data(message_size);
    std::vector<unsigned int> results(num_chunks);

    cuda::device_vector<unsigned char> device_data(message_size);
    cuda::device_vector<unsigned int> device_results(num_chunks);

    for (;;)
    {
      async_read(s, buffer(data), yield);
      make_linked_group(
          copy(command_queue, data.begin(), data.end(), device_data.begin(), deferred),
          checksum_chunks(command_queue, device_data, device_results, deferred),
          reduce_results(command_queue, device_results, deferred),
          copy(command_queue, device_results.begin(), device_results.end(), results.begin(), deferred)
        ).async_wait(yield);
      async_write(s, buffer(std::to_string(static_cast<unsigned int>(results[0])) + "\n"), yield);
    }
  }
  catch (const std::exception&)
  {
  }
}

void listen(tcp::acceptor& a, yield_context yield)
{
  for (;;)
  {
    tcp::socket s{a.get_executor()};
    a.async_accept(s, yield);
    spawn(s.get_executor(),
        [s = std::move(s)](yield_context yield) mutable
        {
          checksum(std::move(s), yield);
        }, detached);
  }
}

int main(int argc, char* argv[])
{
  asio::io_context ctx;
  tcp::acceptor a{ctx, {tcp::v4(), 55555}};
  spawn(ctx.get_executor(), [&](auto yield) { listen(a, yield); }, detached);
  ctx.run();
}
