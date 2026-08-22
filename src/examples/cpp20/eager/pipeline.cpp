//
// pipeline.cpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace asio;
using namespace std::literals::chrono_literals;

// The time taken for a fetch to complete.
constexpr auto fetch_time = 100ms;

// The number of fetches that are kept in flight at any one time.
constexpr int depth = 4;

// The total number of records to process.
constexpr int total = 8;

awaitable<void> pipeline()
{
  auto ex = co_await this_coro::executor;

  std::vector<steady_timer> timers;
  timers.reserve(depth);
  for (int i = 0; i < depth; ++i)
    timers.emplace_back(ex);

  auto async_fetch =
    [&](int slot, int index)
    {
      timers[slot].expires_after(fetch_time);
      return timers[slot].async_wait(append(eager, index));
    };

  // Start the initial fetches. All of them run concurrently.
  using fetch_op = decltype(async_fetch(0, 0));
  std::vector<fetch_op> in_flight;
  in_flight.reserve(depth);
  for (int i = 0; i < depth; ++i)
    in_flight.push_back(async_fetch(i, i));

  for (int i = 0; i < total; ++i)
  {
    const int slot = i % depth;

    auto [ec, index] = co_await std::move(in_flight[slot])(
        as_tuple(use_awaitable));

    if (ec)
      break;

    // Launch a replacement before processing the result, so that the
    // pipeline stays full.
    if (i + depth < total)
      in_flight[slot] = async_fetch(slot, i + depth);

    std::cout << "processed record " << index << "\n";
  }
}

int main()
{
  io_context ctx;

  auto start = std::chrono::steady_clock::now();

  co_spawn(ctx, pipeline(), detached);

  ctx.run();

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);

  std::cout << "processed " << total << " records in "
    << elapsed.count() << "ms\n";

  std::cout << "sequentially this would have taken "
    << std::chrono::duration_cast<std::chrono::milliseconds>(
        total * fetch_time).count() << "ms\n";

  return 0;
}
