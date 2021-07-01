#include <iostream>
#include <asio/detached.hpp>
#include <asio/experimental/coro.hpp>
#include <asio/io_context.hpp>

#include "../unit_test.hpp"

using namespace asio::experimental;

namespace coro
{


asio::experimental::coro<int()> stack_generator(asio::any_io_executor, int i = 1)
{
    for (;;)
    {
        co_yield i;
        i *= 2;
    }
}


asio::experimental::coro<int(int)> stack_accumulate(asio::any_io_executor exec)
{
    auto gen  = stack_generator(exec);
    int offset = 0;
    while (auto next = co_await gen) // 1, 2, 4, 8, ...
        offset  = co_yield *next + offset; // offset is delayed by one cycle
}

asio::experimental::coro<int> main_stack_coro(asio::io_context &, bool & done)
{
    auto g = stack_accumulate(co_await asio::this_coro::executor);

    ASIO_CHECK(g.is_open());
    ASIO_CHECK(1  ==   (co_await g(1000)).value_or(-1));
    ASIO_CHECK(1002 == (co_await g(2000)).value_or(-1));
    ASIO_CHECK(2004 == (co_await g(3000)).value_or(-1));
    ASIO_CHECK(3008 == (co_await g(4000)).value_or(-1));
    ASIO_CHECK(4016 == (co_await g(5000)).value_or(-1));
    ASIO_CHECK(5032 == (co_await g(6000)).value_or(-1));
    ASIO_CHECK(6064 == (co_await g(7000)).value_or(-1));
    ASIO_CHECK(7128 == (co_await g(8000)).value_or(-1));
    ASIO_CHECK(8256 == (co_await g(9000)).value_or(-1));
    ASIO_CHECK(9512 == (co_await g(-1)).value_or(-1));
    done = true;
};


void stack_test()
{
    bool done = false;
    asio::io_context ctx;
    auto k = main_stack_coro(ctx, done);
    k.async_resume(asio::detached);
    ctx.run();
    ASIO_CHECK(done);
}

}


ASIO_TEST_SUITE
(
        "coro",
        ASIO_TEST_CASE(::coro::stack_test)
)