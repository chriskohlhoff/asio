#include <algorithm>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <list>
#include <set>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include "asio.hpp"
#include "chat_message.hpp"

//----------------------------------------------------------------------

typedef std::deque<chat_message> chat_message_queue;

//----------------------------------------------------------------------

class chat_participant
{
public:
  virtual ~chat_participant() {}
  virtual void deliver(const chat_message& msg) = 0;
};

typedef boost::shared_ptr<chat_participant> chat_participant_ptr;

//----------------------------------------------------------------------

class chat_room
{
public:
  void join(chat_participant_ptr participant)
  {
    participants_.insert(participant);
    std::for_each(recent_msgs_.begin(), recent_msgs_.end(),
        boost::bind(&chat_participant::deliver, participant, _1));
  }

  void leave(chat_participant_ptr participant)
  {
    participants_.erase(participant);
  }

  void deliver(const chat_message& msg)
  {
    recent_msgs_.push_back(msg);
    while (recent_msgs_.size() > max_recent_msgs)
      recent_msgs_.pop_front();

    std::for_each(participants_.begin(), participants_.end(),
        boost::bind(&chat_participant::deliver, _1, boost::ref(msg)));
  }

private:
  std::set<chat_participant_ptr> participants_;
  enum { max_recent_msgs = 100 };
  chat_message_queue recent_msgs_;
};

//----------------------------------------------------------------------

class chat_session
  : public chat_participant,
    public boost::enable_shared_from_this<chat_session>
{
public:
  chat_session(asio::demuxer& d, chat_room& r)
    : socket_(d),
      room_(r)
  {
  }

  asio::stream_socket& socket()
  {
    return socket_;
  }

  void start()
  {
    room_.join(shared_from_this());
    asio::async_recv_n(socket_, recv_msg_.data(),
        chat_message::header_length, boost::bind(
          &chat_session::handle_recv_header, shared_from_this(), _1, _2));
  }

  void deliver(const chat_message& msg)
  {
    bool send_in_progress = !send_msgs_.empty();
    send_msgs_.push_back(msg);
    if (!send_in_progress)
    {
      asio::async_send_n(socket_, send_msgs_.front().data(),
          send_msgs_.front().length(), boost::bind(
            &chat_session::handle_send, shared_from_this(), _1, _2));
    }
  }

  void handle_recv_header(const asio::socket_error& error, size_t last_length)
  {
    if (!error && last_length > 0 && recv_msg_.decode_header())
    {
      asio::async_recv_n(socket_, recv_msg_.body(), recv_msg_.body_length(), 
          boost::bind(&chat_session::handle_recv_body, shared_from_this(), _1,
            _2));
    }
    else
    {
      room_.leave(shared_from_this());
    }
  }

  void handle_recv_body(const asio::socket_error& error, size_t last_length)
  {
    if (!error && last_length > 0)
    {
      room_.deliver(recv_msg_);
      asio::async_recv_n(socket_, recv_msg_.data(),
          chat_message::header_length,
          boost::bind(&chat_session::handle_recv_header, shared_from_this(),
            _1, _2));
    }
    else
    {
      room_.leave(shared_from_this());
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
            send_msgs_.front().length(), boost::bind(
              &chat_session::handle_send, shared_from_this(), _1, _2));
      }
    }
    else
    {
      room_.leave(shared_from_this());
    }
  }

private:
  asio::stream_socket socket_;
  chat_room& room_;
  chat_message recv_msg_;
  chat_message_queue send_msgs_;
};

typedef boost::shared_ptr<chat_session> chat_session_ptr;

//----------------------------------------------------------------------

class chat_server
{
public:
  chat_server(asio::demuxer& d, short port)
    : demuxer_(d),
      acceptor_(d, asio::ipv4::address(port))
  {
    chat_session_ptr new_session(new chat_session(demuxer_, room_));
    acceptor_.async_accept(new_session->socket(),
        boost::bind(&chat_server::handle_accept, this, new_session, _1));
  }

  void handle_accept(chat_session_ptr session, const asio::socket_error& error)
  {
    if (!error)
    {
      session->start();
      chat_session_ptr new_session(new chat_session(demuxer_, room_));
      acceptor_.async_accept(new_session->socket(),
          boost::bind(&chat_server::handle_accept, this, new_session, _1));
    }
  }

private:
  asio::demuxer& demuxer_;
  asio::socket_acceptor acceptor_;
  chat_room room_;
};

typedef boost::shared_ptr<chat_server> chat_server_ptr;
typedef std::list<chat_server_ptr> chat_server_list;

//----------------------------------------------------------------------

int main(int argc, char* argv[])
{
  try
  {
    if (argc < 2)
    {
      std::cerr << "Usage: chat_server <port> [<port> ...]\n";
      return 1;
    }

    asio::demuxer d;

    chat_server_list servers;
    for (int i = 1; i < argc; ++i)
    {
      using namespace std; // For atoi.
      chat_server_ptr server(new chat_server(d, atoi(argv[i])));
      servers.push_back(server);
    }

    d.run();
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
