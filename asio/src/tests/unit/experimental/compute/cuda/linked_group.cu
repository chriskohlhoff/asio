#include "asio.hpp"
#include "asio/experimental/deferred.hpp"
#include "asio/experimental/linked_group.hpp"

using asio::experimental::deferred;
using asio::experimental::make_linked_group;

int main()
{
  asio::io_context ctx;

  asio::steady_timer timer1(ctx);
  timer1.expires_after(std::chrono::seconds(1));

  asio::steady_timer timer2(ctx);
  timer2.expires_after(std::chrono::seconds(2));

  bool called = false;
  make_linked_group(
      timer1.async_wait(deferred),
      timer2.async_wait(deferred)
    ).async_wait(
      [&](std::error_code e)
      {
        called = true;
      });

  assert(!called);
  ctx.run();
  assert(called);
}
