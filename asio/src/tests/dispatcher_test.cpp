#include "asio.hpp"
#include <iostream>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

using namespace asio;

void print(int id, int sleep_time, boost::mutex& io_mutex)
{
  boost::mutex::scoped_lock lock(io_mutex);
  std::cout << "Starting " << id << "\n";
  lock.unlock();

  boost::xtime xt;
  boost::xtime_get(&xt, boost::TIME_UTC);
  xt.sec += sleep_time;
  boost::thread::sleep(xt);

  lock.lock();
  std::cout << "Finished " << id << "\n";
  lock.unlock();
}

void inner_print(int id, boost::mutex& io_mutex)
{
  boost::mutex::scoped_lock lock(io_mutex);
  std::cout << "Nested " << id << "\n";
}

void outer_print(demuxer& d, int id, boost::mutex& io_mutex)
{
  boost::mutex::scoped_lock lock(io_mutex);
  std::cout << "Starting " << id << "\n";
  lock.unlock();

  d.operation_immediate(boost::bind(inner_print, id, boost::ref(io_mutex)),
      completion_context::null(), true);

  lock.lock();
  std::cout << "Finished " << id << "\n";
  lock.unlock();
}

void do_dispatch(demuxer& d)
{
  counting_completion_context c1(2);
  counting_completion_context c2(1);
  boost::mutex io_mutex;

  d.operation_immediate(boost::bind(print, 1, 10, boost::ref(io_mutex)), c1);

  d.operation_immediate(boost::bind(print, 2, 5, boost::ref(io_mutex)), c2);

  d.operation_immediate(boost::bind(print, 3, 5, boost::ref(io_mutex)), c1);

  d.operation_immediate(boost::bind(print, 4, 5, boost::ref(io_mutex)), c2);

  d.operation_immediate(boost::bind(print, 5, 5, boost::ref(io_mutex)), c1);

  d.operation_immediate(boost::bind(outer_print, boost::ref(d), 6,
        boost::ref(io_mutex)));

  // Create more threads than the tasks can use, since they are limited by
  // their completion_context counts.
  boost::thread_group threads;
  threads.create_thread(boost::bind(&demuxer::run, &d));
  threads.create_thread(boost::bind(&demuxer::run, &d));
  threads.create_thread(boost::bind(&demuxer::run, &d));
  threads.create_thread(boost::bind(&demuxer::run, &d));
  threads.create_thread(boost::bind(&demuxer::run, &d));
  threads.join_all();
}

int main()
{
  try
  {
    demuxer d;
    do_dispatch(d);
    d.reset();
    do_dispatch(d);
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
