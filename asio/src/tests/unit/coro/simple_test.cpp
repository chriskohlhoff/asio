#include <iostream>
#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/experimental/coro.hpp>
#include <asio/io_context.hpp>
#include <asio/use_awaitable.hpp>

#include <boost/scope_exit.hpp>


#include "../unit_test.hpp"

using namespace asio::experimental;

namespace asio::experimental
{

template struct coro<void(),    void, any_io_executor>;
template struct coro<int(),     void, any_io_executor>;
template struct coro<void(),    int, any_io_executor>;
template struct coro<int(int),  void, any_io_executor>;
template struct coro<int(),     int, any_io_executor>;
template struct coro<int(int),  int, any_io_executor>;

template struct coro<void() noexcept,    void, any_io_executor>;
template struct coro<int() noexcept,     void, any_io_executor>;
template struct coro<void() noexcept,    int, any_io_executor>;
template struct coro<int(int) noexcept,  void, any_io_executor>;
template struct coro<int() noexcept,     int, any_io_executor>;
template struct coro<int(int) noexcept,  int, any_io_executor>;

}

namespace coro
{

asio::experimental::coro<int> generator_impl(asio::any_io_executor , int & last, bool & destroyed)
{
    BOOST_SCOPE_EXIT_ALL(&) {destroyed = true;};
    int i = 0;
    while (true)
        co_yield last = ++i;
}

asio::awaitable<void> generator_test()
{
    int val = 0;
    bool destr = false;
    {
        auto gi = generator_impl(co_await asio::this_coro::executor, val, destr);

        for (int i = 0; i < 10; i++)
        {
            ASIO_CHECK(val == i);
            const auto next = co_await gi.async_resume(asio::use_awaitable);
            ASIO_CHECK(next);
            ASIO_CHECK(val == *next);
            ASIO_CHECK(val == i + 1);
        }

        ASIO_CHECK(!destr);
    }
    ASIO_CHECK(destr);
};

void run_generator_test()
{
    asio::io_context ctx;
    asio::co_spawn(ctx, generator_test, asio::detached);
    ctx.run();
}

asio::experimental::coro<void, int> task_test(asio::io_context &)
{
    co_return 42;
}

asio::experimental::coro<void, int> task_throw(asio::io_context &)
{
    throw std::logic_error(__func__);
    co_return 42;
}


void run_task_test()
{
    asio::io_context ctx;

    bool tt1 = false;
    bool tt2 = false;
    bool tt3 = false;
    bool tt4 = false;
    auto tt = task_test(ctx);
    tt.async_resume(
            [&](std::exception_ptr pt, int i)
            {
                tt1 = true;
                ASIO_CHECK(!pt);
                ASIO_CHECK(i == 42);
                tt.async_resume([&](std::exception_ptr pt, int i) {tt2 = true; ASIO_CHECK(pt);});

            });

    auto tt_2 = task_throw(ctx);

    tt_2.async_resume(
            [&](std::exception_ptr pt, int i)
            {
                tt3 = true;
                ASIO_CHECK(pt);
                tt.async_resume([&](std::exception_ptr pt, int i) {tt4 = true; ASIO_CHECK(pt);});
            });

    ctx.run();

    ASIO_CHECK(tt1);
    ASIO_CHECK(tt2);
    ASIO_CHECK(tt3);
    ASIO_CHECK(tt4);
}

asio::experimental::coro<char> completion_generator_test_impl(asio::any_io_executor , int limit)
{
    for (int i = 0; i < limit; i++)
        co_yield i++;
}

asio::awaitable<void> completion_generator_test()
{
    std::vector<int> res;
    auto g = completion_generator_test_impl(co_await asio::this_coro::executor, 10);

    ASIO_CHECK(g.is_open());
    while (auto val = co_await g.async_resume(asio::use_awaitable))
        res.push_back(*val);

    ASIO_CHECK(!g.is_open());
    ASIO_CHECK((res == std::vector{0,1,2,3,4,5,6,7,8,9}));
};


void run_completion_generator_test()
{
    asio::io_context ctx;
    asio::co_spawn(ctx, completion_generator_test, asio::detached);
    ctx.run();
}

asio::experimental::coro<int(int)> symmetrical_test_impl(asio::any_io_executor )
{
    int i = 0;
    while (true)
        i = (co_yield i) + i;
}

asio::awaitable<void> symmetrical_test()
{
    auto g = symmetrical_test_impl(co_await asio::this_coro::executor);

    ASIO_CHECK(g.is_open());
    ASIO_CHECK(0  == (co_await g.async_resume(0, asio::use_awaitable)).value_or(-1));
    ASIO_CHECK(0  == (co_await g.async_resume(1, asio::use_awaitable)).value_or(-1));
    ASIO_CHECK(1  == (co_await g.async_resume(2, asio::use_awaitable)).value_or(-1));
    ASIO_CHECK(3  == (co_await g.async_resume(3, asio::use_awaitable)).value_or(-1));
    ASIO_CHECK(6  == (co_await g.async_resume(4, asio::use_awaitable)).value_or(-1));
    ASIO_CHECK(10 == (co_await g.async_resume(5, asio::use_awaitable)).value_or(-1));
    ASIO_CHECK(15 == (co_await g.async_resume(6, asio::use_awaitable)).value_or(-1));
    ASIO_CHECK(21 == (co_await g.async_resume(7, asio::use_awaitable)).value_or(-1));
    ASIO_CHECK(28 == (co_await g.async_resume(8, asio::use_awaitable)).value_or(-1));
    ASIO_CHECK(36 == (co_await g.async_resume(9, asio::use_awaitable)).value_or(-1));
};


void run_symmetrical_test()
{
    asio::io_context ctx;
    asio::co_spawn(ctx, symmetrical_test, asio::detached);
    ctx.run();
}

}

ASIO_TEST_SUITE
(
        "coro",
        ASIO_TEST_CASE(::coro::run_generator_test)
            ASIO_TEST_CASE(::coro::run_task_test)
            ASIO_TEST_CASE(::coro::run_symmetrical_test)
            ASIO_TEST_CASE(::coro::run_completion_generator_test)
)
