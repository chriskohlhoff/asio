#include <iostream>
#include <asio/experimental/coro.hpp>
#include <asio/steady_timer.hpp>

#include "../unit_test.hpp"


using namespace asio::experimental;

namespace coro
{

asio::experimental::coro<void(), int> awaiter(asio::any_io_executor exec)
{
    asio::steady_timer timer{exec};
    co_await timer.async_wait(use_coro);
    co_return 42;
}

asio::experimental::coro<void() noexcept, int> awaiter_noexcept(asio::any_io_executor exec)
{
    asio::steady_timer timer{exec};
    auto ec = co_await timer.async_wait(use_coro);
    ASIO_CHECK(ec == std::error_code{});
    co_return 42;
}



void stack_test2()
{
    bool done = false;
    asio::io_context ctx;
    auto k = awaiter(ctx.get_executor());
    auto k2 = awaiter_noexcept(ctx.get_executor());
    k.async_resume([&](std::exception_ptr ex, int res)
            {
                ASIO_CHECK(!ex);
                ASIO_CHECK(res == 42);
                done = true;
            });

    k2.async_resume([&](int res)
           {
               ASIO_CHECK(res == 42);
               done = true;
           });

    ctx.run();
    ASIO_CHECK(done);
}

}

ASIO_TEST_SUITE
(
    "coro",
    ASIO_TEST_CASE(::coro::stack_test2)
)