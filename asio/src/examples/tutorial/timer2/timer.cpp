#include <iostream>
#include "asio.hpp"

void print(const asio::error& /*e*/)
{
  std::cout << "Hello, world!\n";
}

int main()
{
  asio::demuxer d;

  asio::timer t(d, asio::timer::from_now, 5);
  t.async_wait(print);

  d.run();

  return 0;
}
