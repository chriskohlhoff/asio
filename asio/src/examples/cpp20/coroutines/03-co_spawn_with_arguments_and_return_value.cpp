#include <asio.hpp>
#include <iostream>

asio::awaitable<int> func(int a, std::string b)
{
    std::cout << "a = " << a << ", b = \"" << b << "\"\n";
    co_return 233;
}

int main()
{
    /*
        for passing arguments to coroutine,
        call coroutine function directly to get awaitable object,
        then pass it as second argument in co_spawn.

        the future object returned by co_spawn hold the execution result.
    */
    auto future = asio::co_spawn(asio::system_executor{}, func(22, "hello world"), asio::use_future);
    auto result = future.get();
    std::cout << "result from coroutine: " << result << std::endl;
}
