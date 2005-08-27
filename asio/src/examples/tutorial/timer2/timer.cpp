#include <iostream>
#include "asio.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

void print(const asio::error& /*e*/)
{
  std::cout << "Hello, world!\n";
}

int main()
{
  asio::demuxer d;

  asio::deadline_timer t(d, boost::posix_time::seconds(5));
  t.async_wait(print);

  d.run();

  return 0;
}
