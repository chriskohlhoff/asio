#include <iostream>
#include <asio.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

void print(const asio::error& /*e*/,
    asio::deadline_timer* t, int* count)
{
  if (*count < 5)
  {
    std::cout << *count << "\n";
    ++(*count);

    t->expires_at(t->expires_at() + boost::posix_time::seconds(1));
    t->async_wait(boost::bind(print,
          asio::placeholders::error, t, count));
  }
}

int main()
{
  asio::io_service io;

  int count = 0;
  asio::deadline_timer t(io, boost::posix_time::seconds(1));
  t.async_wait(boost::bind(print,
        asio::placeholders::error, &t, &count));

  io.run();

  std::cout << "Final count is " << count << "\n";

  return 0;
}
