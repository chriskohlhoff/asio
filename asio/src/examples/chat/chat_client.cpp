#include <cstdlib>
#include <deque>
#include <iostream>
#include <boost/bind.hpp>
#include "asio.hpp"
#include "chat_message.hpp"

typedef std::deque<chat_message> chat_message_queue;

class chat_client
{
public:
  chat_client(asio::io_service& io_service,
      const asio::ipv4::tcp::endpoint& endpoint)
    : io_service_(io_service),
      socket_(io_service)
  {
    socket_.async_connect(endpoint,
        boost::bind(&chat_client::handle_connect, this,
          asio::placeholders::error));
  }

  void write(const chat_message& msg)
  {
    io_service_.post(boost::bind(&chat_client::do_write, this, msg));
  }

  void close()
  {
    io_service_.post(boost::bind(&chat_client::do_close, this));
  }

private:

  void handle_connect(const asio::error& error)
  {
    if (!error)
    {
      asio::async_read(socket_,
          asio::buffer(read_msg_.data(), chat_message::header_length),
          boost::bind(&chat_client::handle_read_header, this,
            asio::placeholders::error));
    }
  }

  void handle_read_header(const asio::error& error)
  {
    if (!error && read_msg_.decode_header())
    {
      asio::async_read(socket_,
          asio::buffer(read_msg_.body(), read_msg_.body_length()),
          boost::bind(&chat_client::handle_read_body, this,
            asio::placeholders::error));
    }
    else
    {
      do_close();
    }
  }

  void handle_read_body(const asio::error& error)
  {
    if (!error)
    {
      std::cout.write(read_msg_.body(), read_msg_.body_length());
      std::cout << "\n";
      asio::async_read(socket_,
          asio::buffer(read_msg_.data(), chat_message::header_length),
          boost::bind(&chat_client::handle_read_header, this,
            asio::placeholders::error));
    }
    else
    {
      do_close();
    }
  }

  void do_write(chat_message msg)
  {
    bool write_in_progress = !write_msgs_.empty();
    write_msgs_.push_back(msg);
    if (!write_in_progress)
    {
      asio::async_write(socket_,
          asio::buffer(write_msgs_.front().data(),
            write_msgs_.front().length()),
          boost::bind(&chat_client::handle_write, this,
            asio::placeholders::error));
    }
  }

  void handle_write(const asio::error& error)
  {
    if (!error)
    {
      write_msgs_.pop_front();
      if (!write_msgs_.empty())
      {
        asio::async_write(socket_,
            asio::buffer(write_msgs_.front().data(),
              write_msgs_.front().length()),
            boost::bind(&chat_client::handle_write, this,
              asio::placeholders::error));
      }
    }
    else
    {
      do_close();
    }
  }

  void do_close()
  {
    socket_.close();
  }

private:
  asio::io_service& io_service_;
  asio::ipv4::tcp::socket socket_;
  chat_message read_msg_;
  chat_message_queue write_msgs_;
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 3)
    {
      std::cerr << "Usage: chat_client <host> <port>\n";
      return 1;
    }

    asio::io_service io_service;

    using namespace std; // For atoi, strlen and memcpy.
    asio::ipv4::host_resolver hr(io_service);
    asio::ipv4::host h;
    hr.by_name(h, argv[1]);
    asio::ipv4::tcp::endpoint ep(atoi(argv[2]), h.address(0));

    chat_client c(io_service, ep);

    asio::thread t(boost::bind(&asio::io_service::run, &io_service));

    char line[chat_message::max_body_length + 1];
    while (std::cin.getline(line, chat_message::max_body_length + 1))
    {
      chat_message msg;
      msg.body_length(strlen(line));
      memcpy(msg.body(), line, msg.body_length());
      msg.encode_header();
      c.write(msg);
    }

    c.close();
    t.join();
  }
  catch (asio::error& e)
  {
    std::cerr << e << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
