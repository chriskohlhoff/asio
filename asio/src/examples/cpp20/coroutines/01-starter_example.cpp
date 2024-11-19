#include <asio.hpp>
#include <iostream>

/*
    a coroutine func must:
        return an awaitable object
        use co_return instead of return
    
    cpp20 coroutine has no default implemention for awaitable objects, here we use asio's implemention instead.
*/
asio::awaitable<void> simplest_coroutine()
{
    std::cout << "this is a coroutine function.\n";
    co_return;
}

int main()
{
    //create single thread io_context
    asio::io_context context{1};

    /*
        post a coroutine execution to context

        the first argument accepts an asio executor for execution context, here is the io_context object

        call a coroutine function won't execute it but only return an awaitable object,
        pass it as second argument in co_spawn to execute it,
        function pointer or function object returns a awaitable object with no arguments is also allowed here.
        see 03-co_spawn_with_arguments_and_return_value.cpp for more information.
        example:
            asio::co_spawn(context, simplest_coroutine(), asio::detached);

        the third argument asio::detached means we won't wait for result from coroutine,
        if result of coroutine is needed, pass asio::use_future or asio::use_awaitable instead,
        see 02-coroutine_with_system_executor.cpp for more information.
        
        if use asio::use_future, co_spawn function will return a std::future object,
        example:
            auto future = asio::co_spawn(context, simplest_coroutine(), asio::use_future);
            future.get();

        if use asio::use_awaitable, co_spawn function will return an asio::awaitable object,
        which allows co_await, but co_await can only be called in coroutine functions.
        example:
            asio::awaitable<void> function_name()
            {
                auto executor = co_await asio::this_coro::executor;
                co_await asio::co_spawn(executor, **awaitable object**, asio::use_awaitable);
            }
    */

    asio::co_spawn(context, simplest_coroutine(), asio::detached);

    //wait for context to stop
    context.run();
}
