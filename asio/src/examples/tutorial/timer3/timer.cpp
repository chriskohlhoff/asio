#include <iostream>
#include "boost/bind.hpp"
#include "asio.hpp"

void print(asio::timer* t, int* count)
{
  if (*count < 5)
  {
    std::cout << *count << "\n";
    ++(*count);

    t->set(asio::timer::from_existing, 1);
    t->async_wait(boost::bind(print, t, count));
  }
}

int main()
{
  asio::demuxer d;

  int count = 0;
  asio::timer t(d, asio::timer::from_now, 1);
  t.async_wait(boost::bind(print, &t, &count));

  d.run();

  std::cout << "Final count is " << count << "\n";

  return 0;
}
