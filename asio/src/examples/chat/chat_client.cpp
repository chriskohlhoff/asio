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
  chat_client(asio::demuxer& d, const asio::ipv4::tcp::endpoint& endpoint)
    : demuxer_(d),
      socket_(d)
  {
    socket_.async_connect(endpoint,
        boost::bind(&chat_client::handle_connect, this,
          asio::placeholders::error));
  }

  void write(const chat_message& msg)
  {
    demuxer_.post(boost::bind(&chat_client::do_write, this, msg));
  }

  void close()
  {
    demuxer_.post(boost::bind(&chat_client::do_close, this));
  }

private:

  void handle_connect(const asio::error& error)
  {
    if (!error)
    {
      asio::async_read_n(socket_,
          asio::buffers(read_msg_.data(), chat_message::header_length),
          boost::bind(&chat_client::handle_read_header, this,
            asio::placeholders::error,
            asio::placeholders::last_bytes_transferred));
    }
  }

  void handle_read_header(const asio::error& error, size_t last_length)
  {
    if (!error && last_length > 0 && read_msg_.decode_header())
    {
      asio::async_read_n(socket_,
          asio::buffers(read_msg_.body(), read_msg_.body_length()),
          boost::bind(&chat_client::handle_read_body, this,
            asio::placeholders::error,
            asio::placeholders::last_bytes_transferred));
    }
    else
    {
      do_close();
    }
  }

  void handle_read_body(const asio::error& error, size_t last_length)
  {
    if (!error && last_length > 0)
    {
      std::cout.write(read_msg_.body(), read_msg_.body_length());
      std::cout << "\n";
      asio::async_read_n(socket_,
          asio::buffers(read_msg_.data(), chat_message::header_length),
          boost::bind(&chat_client::handle_read_header, this,
            asio::placeholders::error,
            asio::placeholders::last_bytes_transferred));
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
      asio::async_write_n(socket_,
          asio::buffers(write_msgs_.front().data(),
            write_msgs_.front().length()),
          boost::bind(&chat_client::handle_write, this,
            asio::placeholders::error,
            asio::placeholders::last_bytes_transferred));
    }
  }

  void handle_write(const asio::error& error, size_t last_length)
  {
    if (!error && last_length > 0)
    {
      write_msgs_.pop_front();
      if (!write_msgs_.empty())
      {
        asio::async_write_n(socket_,
            asio::buffers(write_msgs_.front().data(),
              write_msgs_.front().length()),
            boost::bind(&chat_client::handle_write, this,
              asio::placeholders::error,
              asio::placeholders::last_bytes_transferred));
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
  asio::demuxer& demuxer_;
  asio::stream_socket socket_;
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

    asio::demuxer d;

    using namespace std; // For atoi, strlen and memcpy.
    asio::ipv4::host_resolver hr(d);
    asio::ipv4::host h;
    hr.get_host_by_name(h, argv[1]);
    asio::ipv4::tcp::endpoint ep(atoi(argv[2]), h.address(0));

    chat_client c(d, ep);

    asio::detail::thread t(boost::bind(&asio::demuxer::run, &d));

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
