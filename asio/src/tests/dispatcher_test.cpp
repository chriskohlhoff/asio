#include "asio.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/thread.hpp"
#include <iostream>
#include <boost/bind.hpp>

using namespace asio;

void print(demuxer& d, int id, int sleep_time, detail::mutex& io_mutex)
{
  detail::mutex::scoped_lock lock(io_mutex);
  std::cout << "Starting " << id << "\n";
  lock.unlock();

  timer t(d, timer::from_now, 5);
  t.wait();

  lock.lock();
  std::cout << "Finished " << id << "\n";
  lock.unlock();
}

void inner_print(int id, detail::mutex& io_mutex)
{
  detail::mutex::scoped_lock lock(io_mutex);
  std::cout << "Nested " << id << "\n";
}

void outer_print(demuxer& d, int id, detail::mutex& io_mutex)
{
  detail::mutex::scoped_lock lock(io_mutex);
  std::cout << "Starting " << id << "\n";
  lock.unlock();

  d.operation_immediate(boost::bind(inner_print, id, boost::ref(io_mutex)),
      null_completion_context(), true);

  lock.lock();
  std::cout << "Finished " << id << "\n";
  lock.unlock();
}

void post_events(demuxer& d, counting_completion_context& c1,
    counting_completion_context& c2, detail::mutex& io_mutex)
{
  // Give all threads an opportunity to start.
  timer t(d, timer::from_now, 2);
  t.wait();

  // Post a bunch of completions to run across the different threads.
  d.operation_immediate(boost::bind(print, boost::ref(d), 1, 10,
        boost::ref(io_mutex)), c1);
  d.operation_immediate(boost::bind(print, boost::ref(d), 2, 5,
        boost::ref(io_mutex)), c2);
  d.operation_immediate(boost::bind(print, boost::ref(d), 3, 5,
        boost::ref(io_mutex)), c1);
  d.operation_immediate(boost::bind(print, boost::ref(d), 4, 5,
        boost::ref(io_mutex)), c2);
  d.operation_immediate(boost::bind(print, boost::ref(d), 5, 5,
        boost::ref(io_mutex)), c1);
  d.operation_immediate(boost::bind(outer_print, boost::ref(d), 6,
        boost::ref(io_mutex)));
}

void do_dispatch(demuxer& d)
{
  counting_completion_context c1(2);
  counting_completion_context c2(1);
  detail::mutex io_mutex;

  d.operation_immediate(boost::bind(post_events, boost::ref(d), boost::ref(c1),
        boost::ref(c2), boost::ref(io_mutex)));

  // Create more threads than the tasks can use, since they are limited by
  // their completion_context counts.
  detail::thread t1(boost::bind(&demuxer::run, &d));
  detail::thread t2(boost::bind(&demuxer::run, &d));
  detail::thread t3(boost::bind(&demuxer::run, &d));
  detail::thread t4(boost::bind(&demuxer::run, &d));
  detail::thread t5(boost::bind(&demuxer::run, &d));
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
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
