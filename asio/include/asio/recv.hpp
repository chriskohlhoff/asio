//
// recv.hpp
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

#ifndef ASIO_RECV_HPP
#define ASIO_RECV_HPP

#include "asio/detail/push_options.hpp"

namespace asio {

/// Read some data from a stream. Returns the number of bytes received or 0 if
/// end-of-file or connection closed. Throws an exception on failure.
template <typename Stream>
size_t recv(Stream& s, void* data, size_t max_length)
{
  return s.recv(data, max_length);
}

/// Start an asynchronous receive. The buffer for the data being received must
/// be valid for the lifetime of the asynchronous operation.
template <typename Stream, typename Handler>
void async_recv(Stream& s, void* data, size_t max_length, Handler handler)
{
  s.async_recv(data, max_length, handler);
}

/// Start an asynchronous receive. The buffer for the data being received must
/// be valid for the lifetime of the asynchronous operation.
template <typename Stream, typename Handler, typename Completion_Context>
void async_recv(Stream& s, void* data, size_t max_length, Handler handler,
    Completion_Context& context)
{
  s.async_recv(data, max_length, handler, context);
}

/// Read the specified amount of data from the stream before returning. Returns
/// the number of bytes received on the last recv operation or 0 if end-of-file
/// or connection closed. Throws an exception on failure.
template <typename Stream>
size_t recv_n(Stream& s, void* data, size_t length,
    size_t* total_bytes_recvd = 0)
{
  int bytes_recvd = 0;
  size_t total_recvd = 0;
  while (total_recvd < length)
  {
    bytes_recvd = recv(s, static_cast<char*>(data) + total_recvd,
        length - total_recvd);
    if (bytes_recvd == 0)
    {
      if (total_bytes_recvd)
        *total_bytes_recvd = total_recvd;
      return bytes_recvd;
    }
    total_recvd += bytes_recvd;
  }
  if (total_bytes_recvd)
    *total_bytes_recvd = total_recvd;
  return bytes_recvd;
}

namespace detail
{
  template <typename Stream, typename Handler, typename Completion_Context>
  class recv_n_handler
  {
  public:
    recv_n_handler(Stream& stream, void* data, size_t length, Handler handler,
        Completion_Context& context)
      : stream_(stream),
        data_(data),
        length_(length),
        total_recvd_(0),
        handler_(handler),
        context_(context)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_recvd)
    {
      total_recvd_ += bytes_recvd;
      if (e || bytes_recvd == 0 || total_recvd_ == length_)
      {
        handler_(e, total_recvd_, bytes_recvd);
      }
      else
      {
        async_recv(stream_, static_cast<char*>(data_) + total_recvd_,
            length_ - total_recvd_, *this, context_);
      }
    }

  private:
    Stream& stream_;
    void* data_;
    size_t length_;
    size_t total_recvd_;
    Handler handler_;
    Completion_Context& context_;
  };
} // namespace detail

/// Start an asynchronous receive that will not complete until the specified
/// amount of data has been received. The target buffer for the data being
/// received must be valid for the lifetime of the asynchronous operation.
template <typename Stream, typename Handler>
void async_recv_n(Stream& s, void* data, size_t length, Handler handler)
{
  async_recv(s, data, length,
      detail::recv_n_handler<Stream, Handler, null_completion_context>(s, data,
        length, handler, null_completion_context::instance()));
}

/// Start an asynchronous receive that will not complete until the specified
/// amount of data has been received. The target buffer for the data being
/// received must be valid for the lifetime of the asynchronous operation.
template <typename Stream, typename Handler, typename Completion_Context>
void async_recv_n(Stream& s, void* data, size_t length, Handler handler,
    Completion_Context& context)
{
  async_recv(s, data, length,
      detail::recv_n_handler<Stream, Handler, Completion_Context>(s, data,
        length, handler, context), context);
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_RECV_HPP
