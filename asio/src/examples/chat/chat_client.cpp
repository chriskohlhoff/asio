#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <string>
#include <boost/bind.hpp>
#include "asio.hpp"
#include "chat_message.hpp"

typedef std::list<chat_message> chat_message_list;

class chat_client
{
public:
  chat_client(asio::demuxer& d, short port, const std::string& host)
    : demuxer_(d),
      connector_(d),
      socket_(d)
  {
    connector_.async_connect(socket_, asio::inet_address_v4(port, host),
        boost::bind(&chat_client::handle_connect, this, _1));
  }

  void send(const chat_message& msg)
  {
    demuxer_.operation_immediate(
        boost::bind(&chat_client::do_send, this, msg));
  }

  void close()
  {
    demuxer_.operation_immediate(boost::bind(&chat_client::do_close, this));
  }

private:

  void handle_connect(const asio::socket_error& error)
  {
    if (!error)
    {
      async_recv_chat_message(socket_, recv_msg_,
          boost::bind(&chat_client::handle_recv, this, _1, _2, _3));
    }
  }

  void handle_recv(const asio::socket_error& error, size_t length,
      size_t last_length)
  {
    if (!error && last_length > 0)
    {
      std::cout.write(recv_msg_.body(), recv_msg_.body_length());
      std::cout << "\n";
      async_recv_chat_message(socket_, recv_msg_,
          boost::bind(&chat_client::handle_recv, this, _1, _2, _3));
    }
  }

  void do_send(chat_message msg)
  {
    bool send_in_progress = !send_msgs_.empty();
    send_msgs_.push_back(msg);
    if (!send_in_progress)
    {
      async_send_chat_message(socket_, send_msgs_.front(),
          boost::bind(&chat_client::handle_send, this, _1, _2, _3));
    }
  }

  void handle_send(const asio::socket_error& error, size_t length,
      size_t last_length)
  {
    if (!error && last_length > 0)
    {
      send_msgs_.pop_front();
      if (!send_msgs_.empty())
      {
        async_send_chat_message(socket_, send_msgs_.front(),
            boost::bind(&chat_client::handle_send, this, _1, _2, _3));
      }
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
  chat_message_list send_msgs_;
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

    using namespace std; // For atoi, strlen, strcpy and sprintf.
    chat_client c(d, atoi(argv[2]), argv[1]);

    asio::detail::thread t(boost::bind(&asio::demuxer::run, &d));

    char line[chat_message::max_body_length + 1];
    while (std::cin.getline(line, chat_message::max_body_length + 1))
    {
      chat_message msg;
      msg.length(chat_message::header_length + strlen(line));
      sprintf(msg.data(), "%4d", msg.body_length());
      strncpy(msg.body(), line, msg.body_length());
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
