#include <asio/io_service.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/spawn.hpp>
#include <asio/steady_timer.hpp>
#include <asio/write.hpp>
#include <memory>

using asio::ip::tcp;

class session : public std::enable_shared_from_this<session>
{
public:
  explicit session(tcp::socket socket)
    : strand_(socket.get_io_service()),
      socket_(std::move(socket)),
      timer_(strand_.get_io_service())
  {
  }

  void go()
  {
    asio::spawn(strand_,
        std::bind(&session::echo,
          shared_from_this(), std::placeholders::_1));
    asio::spawn(strand_,
        std::bind(&session::timeout,
          shared_from_this(), std::placeholders::_1));
  }

private:
  void echo(asio::yield_context yield)
  {
    try
    {
      char data[128];
      for (;;)
      {
        timer_.expires_from_now(std::chrono::seconds(10));
        std::size_t n = socket_.async_read_some(asio::buffer(data), yield);
        asio::async_write(socket_, asio::buffer(data, n), yield);
      }
    }
    catch (std::exception& e)
    {
      socket_.close();
      timer_.cancel();
    }
  }

  void timeout(asio::yield_context yield)
  {
    while (socket_.is_open())
    {
      std::error_code ignored_ec;
      timer_.async_wait(yield(ignored_ec));
      if (timer_.expires_from_now() <= std::chrono::seconds(0))
        socket_.close();
    }
  }

  asio::io_service::strand strand_;
  tcp::socket socket_;
  asio::steady_timer timer_;
};

int main()
{
  asio::io_service io_service;

  asio::spawn(io_service,
      [&](asio::yield_context yield)
      {
        tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), 55555));
        for (;;)
        {
          std::error_code ec;
          tcp::socket socket(io_service);
          acceptor.async_accept(socket, yield(ec));
          if (!ec) std::make_shared<session>(std::move(socket))->go();
        }
      });

  io_service.run();
}
