#include <iostream>
#include "asio.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

int main()
{
  asio::demuxer d;

  asio::deadline_timer t(d, boost::posix_time::seconds(5));
  t.wait();

  std::cout << "Hello, world!\n";

  return 0;
}
