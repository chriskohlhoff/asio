#ifndef CHAT_MESSAGE_HPP
#define CHAT_MESSAGE_HPP

#include <vector>
#include <strstream>
#include <boost/bind.hpp>
#include "asio.hpp"

class chat_message
{
public:
  enum { header_length = 4 };
  enum { max_body_length = 512 };

  const char* data() const
  {
    return &data_[0];
  }

  char* data()
  {
    return &data_[0];
  }

  size_t length() const
  {
    return data_.size();
  }

  void length(size_t l)
  {
    data_.resize(l);
  }

  const char* body() const
  {
    return &data_[0] + header_length;
  }

  char* body()
  {
    return &data_[0] + header_length;
  }

  size_t body_length() const
  {
    return data_.size() - header_length;
  }

private:
  std::vector<char> data_;
};

template <typename Stream>
size_t send_chat_message(Stream& s, chat_message& msg)
{
  return asio::send_n(s, msg.data(), msg.length());
}

template <typename Stream, typename Handler>
void async_send_chat_message(Stream& s, chat_message& msg, Handler handler)
{
  asio::async_send_n(s, msg.data(), msg.length(), handler);
}

template <typename Stream>
size_t recv_chat_message(Stream& s, chat_message& msg)
{
  msg.length(chat_message::header_length);
  if (asio::recv_n(s, msg.data(), chat_message::header_length) == 0)
    return 0;
  std::istrstream is(msg.data(), chat_message::header_length);
  size_t body_length = 0;
  is >> body_length;
  if (!is || body_length > chat_message::max_body_length)
    return 0;
  msg.length(chat_message::header_length + body_length);
  return asio::recv_n(s, msg.body(), msg.body_length());
}

template <typename Stream, typename Handler>
class recv_chat_message_handler
{
public:
  recv_chat_message_handler(Stream& stream, chat_message& msg, Handler handler)
    : stream_(stream),
      msg_(msg),
      handler_(handler)
  {
  }

  template <typename Error>
  void operator()(const Error& error, size_t length, size_t last_length)
  {
    if (!error && last_length > 0)
    {
      std::istrstream is(msg_.data(), chat_message::header_length);
      size_t body_length = 0;
      is >> body_length;
      if (is && body_length <= chat_message::max_body_length)
      {
        msg_.length(chat_message::header_length + body_length);
        asio::async_recv_n(stream_, msg_.body(), msg_.body_length(), handler_);
      }
      else
      {
        last_length = 0;
        handler_(error, length, last_length);
      }
    }
    else
    {
      handler_(error, length, last_length);
    }
  }

private:
  Stream& stream_;
  chat_message& msg_;
  Handler handler_;
};

template <typename Stream, typename Handler>
void async_recv_chat_message(Stream& s, chat_message& msg, Handler handler)
{
  msg.length(chat_message::header_length);
  asio::async_recv_n(s, msg.data(), chat_message::header_length,
      recv_chat_message_handler<Stream, Handler>(s, msg, handler));
}

#endif // CHAT_MESSAGE_HPP
