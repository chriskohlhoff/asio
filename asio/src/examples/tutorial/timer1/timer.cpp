#include <iostream>
#include "asio.hpp"

int main()
{
  asio::demuxer d;

  asio::timer t(d, asio::time::now() + 5);
  t.wait();

  std::cout << "Hello, world!\n";

  return 0;
}
