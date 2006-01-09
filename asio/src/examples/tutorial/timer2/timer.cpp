#include <iostream>
#include <asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

void print(const asio::error& /*e*/)
{
  std::cout << "Hello, world!\n";
}

int main()
{
  asio::io_service io;

  asio::deadline_timer t(io, boost::posix_time::seconds(5));
  t.async_wait(print);

  io.run();

  return 0;
}
