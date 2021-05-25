#include <asio.hpp>
#include <chrono>
#include <iostream>

asio::awaitable<void> simple_coroutine_func()
{
    std::cout << "in coroutine function.\n";
    co_return;
}

int main()
{
    /*
        according to doc, The system executor represents an execution context where 
        functions are permitted to run on arbitrary threads.

        coroutines scheduled in system_executor will be canceled if main ended.

        here we pass asio::use_future as the third argument to get a std::future object for coroutine.
    */
    auto future = asio::co_spawn(asio::system_executor{}, simple_coroutine_func, asio::use_future);
    std::cout << "wait for coroutine...\n";
    future.get(); //wait for coroutine execution by future object.
    std::cout << "after coroutine stopped.\n";
}
