#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>

using namespace asio;

void print(int* counter)
{
  std::cout << "Timeout " << (*counter)++ << "\n";
}

int main()
{
  try
  {
    demuxer d;
    timer_queue tq(d);

    int counter = 0;
    boost::xtime print_time;
    boost::xtime_get(&print_time, boost::TIME_UTC);
    print_time.sec += 1;
    boost::xtime print_interval;
    print_interval.sec = 1;
    print_interval.nsec = 0;
    int timer_id = tq.schedule_timer(print_time, print_interval,
        boost::bind(print, &counter));

    boost::xtime cancel_time;
    boost::xtime_get(&cancel_time, boost::TIME_UTC);
    cancel_time.sec += 6;
    cancel_time.nsec = 1000000;
    tq.schedule_timer(cancel_time,
        boost::bind(&timer_queue::cancel_timer, &tq, timer_id));

    d.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
