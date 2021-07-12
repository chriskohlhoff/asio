#include <asio.hpp>
#include <iostream>

asio::awaitable<int> function1()
{
    std::cout << "function 1 start.\n";
    co_return 233;
}

asio::awaitable<int> function2()
{
    std::cout << "function 2 start.\n";
    /*
        use co_await in coroutine to wait for another coroutine function execution.
    */
    int result = co_await function1();
    std::cout << "after function 1.\n";
    result += 233;
    co_return result;
}

int main()
{
    auto future = asio::co_spawn(asio::system_executor{}, function2, asio::use_future);
    auto result = future.get();
    std::cout << "result from function2: " << result << "\n";
}