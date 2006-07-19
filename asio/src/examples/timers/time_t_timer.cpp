#include <asio.hpp>
#include <ctime>
#include <iostream>

struct time_t_traits
{
  // The time type.
  struct time_type
  {
    time_type() : value(0) {}
    time_type(std::time_t v) : value(v) {}
    std::time_t value;
  };

  // The duration type.
  struct duration_type
  {
    duration_type() : value(0) {}
    duration_type(std::time_t v) : value(v) {}
    std::time_t value;
  };

  // Get the current time.
  static time_type now()
  {
    return std::time(0);
  }

  // Add a duration to a time.
  static time_type add(const time_type& t, const duration_type& d)
  {
    return t.value + d.value;
  }

  // Subtract one time from another.
  static duration_type subtract(const time_type& t1, const time_type& t2)
  {
    return duration_type(t1.value - t2.value);
  }

  // Test whether one time is less than another.
  static bool less_than(const time_type& t1, const time_type& t2)
  {
    return t1.value < t2.value;
  }

  // Convert to POSIX duration type.
  static boost::posix_time::time_duration to_posix_duration(
      const duration_type& d)
  {
    return boost::posix_time::seconds(d.value);
  }
};

typedef asio::basic_deadline_timer<
    std::time_t, time_t_traits> time_t_timer;

void handle_timeout(const asio::error&)
{
  std::cout << "handle_timeout\n";
}

int main()
{
  try
  {
    asio::io_service io_service;

    time_t_timer timer(io_service);

    timer.expires_from_now(5);
    std::cout << "Starting synchronous wait\n";
    timer.wait();
    std::cout << "Finished synchronous wait\n";

    timer.expires_from_now(5);
    std::cout << "Starting asynchronous wait\n";
    timer.async_wait(handle_timeout);
    io_service.run();
    std::cout << "Finished asynchronous wait\n";
  }
  catch (std::exception& e)
  {
    std::cout << "Exception: " << e.what() << "\n";
  }

  return 0;
}
