#include <iostream>
#include "boost/bind.hpp"
#include "asio.hpp"

class printer
{
public:
  printer(asio::demuxer& d)
    : timer_(d, asio::timer::from_now, 1),
      count_(0)
  {
    timer_.async_wait(boost::bind(&printer::print, this));
  }

  ~printer()
  {
    std::cout << "Final count is " << count_ << "\n";
  }

  void print()
  {
    if (count_ < 5)
    {
      std::cout << count_ << "\n";
      ++count_;

      timer_.set(asio::timer::from_existing, 1);
      timer_.async_wait(boost::bind(&printer::print, this));
    }
  }

private:
  asio::timer timer_;
  int count_;
};

int main()
{
  asio::demuxer d;
  printer p(d);
  d.run();

  return 0;
}
