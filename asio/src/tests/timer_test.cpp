#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>

using namespace asio;

void print(timer* print_timer, int* counter, bool* cancelled)
{
  if (!*cancelled)
  {
    std::cout << "Timeout " << (*counter)++ << "\n";
    print_timer->set(timer::from_existing, 1);
    print_timer->async_wait(
        boost::bind(print, print_timer, counter, cancelled));
  }
}

void cancel(timer* print_timer, bool* cancelled)
{
  *cancelled = true;
  print_timer->expire();
}

int main()
{
  try
  {
    demuxer d;
    int counter = 0;
    bool cancelled = false;

    timer print_timer(d, timer::from_now, 1);
    print_timer.async_wait(
        boost::bind(print, &print_timer, &counter, &cancelled));

    timer cancel_timer(d, timer::from_now, 6, 100000);
    cancel_timer.async_wait(boost::bind(cancel, &print_timer, &cancelled));

    d.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
