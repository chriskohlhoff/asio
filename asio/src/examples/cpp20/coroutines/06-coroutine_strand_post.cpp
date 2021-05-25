#include <asio.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <latch>

using namespace std::chrono_literals;

int main()
{
    /*
        use make_strand function to make a strand from system_executor.

        for more information about strand,
        please go https://think-async.com/Asio/asio-1.18.2/doc/asio/overview/core/strands.html for help
    */
    auto outputStrand = asio::make_strand(asio::system_executor{});
    std::latch work_done{10};

    auto func = [&outputStrand, &work_done](int i) -> asio::awaitable<void> {
        /*
            replace post callback with asio::use_awaitable to enable coroutine support for asio::post
            the code following this line is equivalent to callback function passed into asio::post
        */
        co_await asio::post(outputStrand, asio::use_awaitable);

        //the following code will be executed by outputStrand and be scheduled as if in the same thread.
        std::cout << "current task: " << i << std::endl;
        std::this_thread::sleep_for(3s); //simulates long compute task
        std::cout << "task " << i << " ended.\n";
        work_done.count_down(); //notify work finished.
        co_return;
    };

    //create 10 task and schedule as coroutine
    for (int i = 0; i < 10; i++)
    {
        asio::co_spawn(asio::system_executor{}, func(i), asio::detached);
    }

    //wait for all coroutine finished
    work_done.wait();
}
