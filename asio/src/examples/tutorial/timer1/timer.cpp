#include <iostream>
#include "asio.hpp"

int main()
{
  asio::demuxer d;

  asio::timer t(d, asio::timer::from_now, 5);
  t.wait();

  std::cout << "Hello, world!\n";

  return 0;
}
