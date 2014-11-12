#include <asio/ts/executor.hpp>
#include <asio/ts/thread_pool.hpp>
#include <iostream>
#include <string>

using asio::dispatch;
using asio::make_work;
using asio::post;
using asio::thread_pool;
using asio::wrap;

// A function to asynchronously read a single line from an input stream.
template <class Handler>
void async_getline(std::istream& is, Handler handler)
{
  // Create executor_work for the handler's associated executor.
  auto work = make_work(handler);

  // Post a function object to do the work asynchronously.
  post([&is, work, handler=std::move(handler)]() mutable
      {
        std::string line;
        std::getline(is, line);

        // Pass the result to the handler, via the associated executor.
        dispatch(work.get_executor(),
            [line=std::move(line), handler=std::move(handler)]() mutable
            {
              handler(std::move(line));
            });
      });
}

int main()
{
  thread_pool pool;

  std::cout << "Enter a line: ";

  async_getline(std::cin,
      wrap(pool, [](std::string line)
        {
          std::cout << "Line: " << line << "\n";
        }));

  pool.join();
}
