//
// send.hpp
// ~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_SEND_HPP
#define ASIO_SEND_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/bind_handler.hpp"

namespace asio {

/// Write some data to a stream. Returns the number of bytes sent or 0 if
/// end-of-file or connection closed. Throws an exception on failure.
template <typename Stream>
size_t send(Stream& s, const void* data, size_t length)
{
  return s.send(data, length);
}

/// Start an asynchronous send. The buffer containing the data being sent
/// must be valid for the lifetime of the asynchronous operation.
template <typename Stream, typename Handler>
void async_send(Stream& s, const void* data, size_t length, Handler handler)
{
  s.async_send(data, length, handler);
}

/// Start an asynchronous send. The buffer containing the data being sent
/// must be valid for the lifetime of the asynchronous operation.
template <typename Stream, typename Handler, typename Completion_Context>
void async_send(Stream& s, const void* data, size_t length, Handler handler,
    Completion_Context& context)
{
  s.async_send(data, length, handler, context);
}

/// Write all of the given data to the stream before returning. Returns the
/// number of bytes sent on the last send or 0 if end-of-file or connection
/// closed. Throws an exception on failure.
template <typename Stream>
size_t send_n(Stream& s, const void* data, size_t length,
    size_t* total_bytes_sent = 0)
{
  int bytes_sent = 0;
  size_t total_sent = 0;
  while (total_sent < length)
  {
    bytes_sent = send(s, static_cast<const char*>(data) + total_sent,
        length - total_sent);
    if (bytes_sent == 0)
    {
      if (total_bytes_sent)
        *total_bytes_sent = total_sent;
      return bytes_sent;
    }
    total_sent += bytes_sent;
  }
  if (total_bytes_sent)
    *total_bytes_sent = total_sent;
  return bytes_sent;
}

namespace detail
{
#if defined(_MSC_VER)
  static void send_n_optimiser_bug_workaround() {}
#endif // _MSC_VER

  template <typename Stream, typename Handler, typename Completion_Context>
  class send_n_handler
  {
  public:
    send_n_handler(Stream& stream, const void* data, size_t length,
        Handler handler, Completion_Context& context)
      : stream_(stream),
        data_(data),
        length_(length),
        total_sent_(0),
        handler_(handler),
        context_(context)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_sent)
    {
      total_sent_ += bytes_sent;
      if (e || bytes_sent == 0 || total_sent_ == length_)
      {
#if defined(_MSC_VER)
        // Unless we put this function call here, the MSVC6 optimiser totally
        // removes this function (incorrectly of course) and async_send_n calls
        // may not work correctly.
        send_n_optimiser_bug_workaround();
#endif // _MSC_VER

        stream_.demuxer().operation_immediate(detail::bind_handler(handler_, e,
              total_sent_, bytes_sent), context_, true);
      }
      else
      {
        async_send(stream_, static_cast<const char*>(data_) + total_sent_,
            length_ - total_sent_, *this);
      }
    }

  private:
    Stream& stream_;
    const void* data_;
    size_t length_;
    size_t total_sent_;
    Handler handler_;
    Completion_Context& context_;
  };
} // namespace detail

/// Start an asynchronous send that will not complete until the specified
/// amount of data has been sent. The buffer containing the data being sent
/// must be valid for the lifetime of the asynchronous operation.
template <typename Stream, typename Handler>
void async_send_n(Stream& s, const void* data, size_t length, Handler handler)
{
  async_send(s, data, length,
      detail::send_n_handler<Stream, Handler, null_completion_context>(s,
        data, length, handler, null_completion_context::instance()));
}

/// Start an asynchronous send that will not complete until the specified
/// amount of data has been sent. The buffer containing the data being sent
/// must be valid for the lifetime of the asynchronous operation.
template <typename Stream, typename Handler, typename Completion_Context>
void async_send_n(Stream& s, const void* data, size_t length, Handler handler,
    Completion_Context& context)
{
  async_send(s, data, length,
      detail::send_n_handler<Stream, Handler, Completion_Context>(s, data,
        length, handler, context));
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SEND_HPP
