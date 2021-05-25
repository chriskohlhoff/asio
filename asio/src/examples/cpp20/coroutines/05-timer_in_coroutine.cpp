#include <asio.hpp>
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

asio::awaitable<void> wait_five_seconds()
{
    auto executor = co_await asio::this_coro::executor;
    asio::steady_timer timer(executor, 5s);
    /*
        in origin asio, async_wait receives a callback,
        the callback passed in will be excuted when timer expires.

        but with coroutine support, we can replace callback with asio::use_awaitable,
        and use co_await to suspend coroutine until timer expire.
        code following this line will be executed after timer expired.
    */
    co_await timer.async_wait(asio::use_awaitable);
    std::cout << "timer expired.\n";
}

int main()
{
    asio::io_context context{1};
    asio::co_spawn(context, wait_five_seconds, asio::detached);
    context.run();
}
