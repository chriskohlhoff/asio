#include <iostream>
#include "boost/bind.hpp"
#include "asio.hpp"

void print(const asio::error& /*e*/, asio::timer* t, int* count)
{
  if (*count < 5)
  {
    std::cout << *count << "\n";
    ++(*count);

    t->expiry(t->expiry() + 1);
    t->async_wait(boost::bind(print, asio::arg::error, t, count));
  }
}

int main()
{
  asio::demuxer d;

  int count = 0;
  asio::timer t(d, asio::time::now() + 1);
  t.async_wait(boost::bind(print, asio::arg::error, &t, &count));

  d.run();

  std::cout << "Final count is " << count << "\n";

  return 0;
}
