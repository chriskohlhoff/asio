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
  chat_client(asio::demuxer& d, short port, const char* host)
    : demuxer_(d),
      connector_(d),
      socket_(d)
  {
    connector_.async_connect(socket_, asio::ipv4::address(port, host),
        boost::bind(&chat_client::handle_connect, this, asio::arg::error));
  }

  void send(const chat_message& msg)
  {
    demuxer_.post(boost::bind(&chat_client::do_send, this, msg));
  }

  void close()
  {
    demuxer_.post(boost::bind(&chat_client::do_close, this));
  }

private:

  void handle_connect(const asio::socket_error& error)
  {
    if (!error)
    {
      asio::async_recv_n(socket_, recv_msg_.data(),
          chat_message::header_length,
          boost::bind(&chat_client::handle_recv_header, this, asio::arg::error,
            asio::arg::last_bytes_recvd));
    }
  }

  void handle_recv_header(const asio::socket_error& error, size_t last_length)
  {
    if (!error && last_length > 0 && recv_msg_.decode_header())
    {
      asio::async_recv_n(socket_, recv_msg_.body(), recv_msg_.body_length(), 
          boost::bind(&chat_client::handle_recv_body, this, asio::arg::error,
            asio::arg::last_bytes_recvd));
    }
    else
    {
      do_close();
    }
  }

  void handle_recv_body(const asio::socket_error& error, size_t last_length)
  {
    if (!error && last_length > 0)
    {
      std::cout.write(recv_msg_.body(), recv_msg_.body_length());
      std::cout << "\n";
      asio::async_recv_n(socket_, recv_msg_.data(),
          chat_message::header_length,
          boost::bind(&chat_client::handle_recv_header, this,
            asio::arg::error, asio::arg::last_bytes_recvd));
    }
    else
    {
      do_close();
    }
  }

  void do_send(chat_message msg)
  {
    bool send_in_progress = !send_msgs_.empty();
    send_msgs_.push_back(msg);
    if (!send_in_progress)
    {
      asio::async_send_n(socket_, send_msgs_.front().data(),
          send_msgs_.front().length(),
          boost::bind(&chat_client::handle_send, this, asio::arg::error,
            asio::arg::last_bytes_sent));
    }
  }

  void handle_send(const asio::socket_error& error, size_t last_length)
  {
    if (!error && last_length > 0)
    {
      send_msgs_.pop_front();
      if (!send_msgs_.empty())
      {
        asio::async_send_n(socket_, send_msgs_.front().data(),
            send_msgs_.front().length(),
            boost::bind(&chat_client::handle_send, this, asio::arg::error,
              asio::arg::last_bytes_sent));
      }
    }
    else
    {
      do_close();
    }
  }

  void do_close()
  {
    connector_.close();
    socket_.close();
  }

private:
  asio::demuxer& demuxer_;
  asio::socket_connector connector_;
  asio::stream_socket socket_;
  chat_message recv_msg_;
  chat_message_queue send_msgs_;
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
    chat_client c(d, atoi(argv[2]), argv[1]);

    asio::detail::thread t(boost::bind(&asio::demuxer::run, &d));

    char line[chat_message::max_body_length + 1];
    while (std::cin.getline(line, chat_message::max_body_length + 1))
    {
      chat_message msg;
      msg.body_length(strlen(line));
      memcpy(msg.body(), line, msg.body_length());
      msg.encode_header();
      c.send(msg);
    }

    c.close();
    t.join();
  }
  catch (asio::socket_error& e)
  {
    std::cerr << "Socket error: " << e.message() << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
