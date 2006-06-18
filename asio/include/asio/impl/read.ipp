//
// read.ipp
// ~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_READ_IPP
#define ASIO_READ_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/buffer.hpp"
#include "asio/completion_condition.hpp"
#include "asio/error_handler.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/consuming_buffers.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"

namespace asio {

template <typename Sync_Read_Stream, typename Mutable_Buffers,
    typename Completion_Condition, typename Error_Handler>
std::size_t read(Sync_Read_Stream& s, const Mutable_Buffers& buffers,
    Completion_Condition completion_condition, Error_Handler error_handler)
{
  asio::detail::consuming_buffers<
    mutable_buffer, Mutable_Buffers> tmp(buffers);
  std::size_t total_transferred = 0;
  while (tmp.begin() != tmp.end())
  {
    typename Sync_Read_Stream::error_type e;
    std::size_t bytes_transferred = s.read_some(tmp, assign_error(e));
    tmp.consume(bytes_transferred);
    total_transferred += bytes_transferred;
    if (completion_condition(e, total_transferred))
    {
      error_handler(e);
      return total_transferred;
    }
  }
  typename Sync_Read_Stream::error_type e;
  error_handler(e);
  return total_transferred;
}

template <typename Sync_Read_Stream, typename Mutable_Buffers>
inline std::size_t read(Sync_Read_Stream& s, const Mutable_Buffers& buffers)
{
  return read(s, buffers, transfer_all(), throw_error());
}

template <typename Sync_Read_Stream, typename Mutable_Buffers,
    typename Completion_Condition>
inline std::size_t read(Sync_Read_Stream& s, const Mutable_Buffers& buffers,
    Completion_Condition completion_condition)
{
  return read(s, buffers, completion_condition, throw_error());
}

template <typename Sync_Read_Stream, typename Allocator,
    typename Completion_Condition, typename Error_Handler>
std::size_t read(Sync_Read_Stream& s,
    asio::basic_streambuf<Allocator>& b,
    Completion_Condition completion_condition, Error_Handler error_handler)
{
  std::size_t total_transferred = 0;
  for (;;)
  {
    typename Sync_Read_Stream::error_type e;
    std::size_t bytes_transferred = s.read_some(
        b.prepare(512), assign_error(e));
    b.commit(bytes_transferred);
    total_transferred += bytes_transferred;
    if (completion_condition(e, total_transferred))
    {
      error_handler(e);
      return total_transferred;
    }
  }
  typename Sync_Read_Stream::error_type e;
  error_handler(e);
  return total_transferred;
}

template <typename Sync_Read_Stream, typename Allocator>
inline std::size_t read(Sync_Read_Stream& s,
    asio::basic_streambuf<Allocator>& b)
{
  return read(s, b, transfer_all(), throw_error());
}

template <typename Sync_Read_Stream, typename Allocator,
    typename Completion_Condition>
inline std::size_t read(Sync_Read_Stream& s,
    asio::basic_streambuf<Allocator>& b,
    Completion_Condition completion_condition)
{
  return read(s, b, completion_condition, throw_error());
}

namespace detail
{
  template <typename Async_Read_Stream, typename Mutable_Buffers,
      typename Completion_Condition, typename Handler>
  class read_handler
  {
  public:
    read_handler(Async_Read_Stream& stream, const Mutable_Buffers& buffers,
        Completion_Condition completion_condition, Handler handler)
      : stream_(stream),
        buffers_(buffers),
        total_transferred_(0),
        completion_condition_(completion_condition),
        handler_(handler)
    {
    }

    void operator()(const typename Async_Read_Stream::error_type& e,
        std::size_t bytes_transferred)
    {
      total_transferred_ += bytes_transferred;
      buffers_.consume(bytes_transferred);
      if (completion_condition_(e, total_transferred_)
          || buffers_.begin() == buffers_.end())
      {
        stream_.io_service().dispatch(
            detail::bind_handler(handler_, e, total_transferred_));
      }
      else
      {
        stream_.async_read_some(buffers_, *this);
      }
    }

    friend void* asio_handler_allocate(std::size_t size,
        read_handler<Async_Read_Stream, Mutable_Buffers,
          Completion_Condition, Handler>* this_handler)
    {
      return asio_handler_alloc_helpers::allocate(
          size, &this_handler->handler_);
    }

    friend void asio_handler_deallocate(void* pointer, std::size_t size,
        read_handler<Async_Read_Stream, Mutable_Buffers,
          Completion_Condition, Handler>* this_handler)
    {
      asio_handler_alloc_helpers::deallocate(
          pointer, size, &this_handler->handler_);
    }

  private:
    Async_Read_Stream& stream_;
    asio::detail::consuming_buffers<
      mutable_buffer, Mutable_Buffers> buffers_;
    std::size_t total_transferred_;
    Completion_Condition completion_condition_;
    Handler handler_;
  };
} // namespace detail

template <typename Async_Read_Stream, typename Mutable_Buffers,
    typename Completion_Condition, typename Handler>
inline void async_read(Async_Read_Stream& s, const Mutable_Buffers& buffers,
    Completion_Condition completion_condition, Handler handler)
{
  s.async_read_some(buffers,
      detail::read_handler<Async_Read_Stream, Mutable_Buffers,
        Completion_Condition, Handler>(
          s, buffers, completion_condition, handler));
}

template <typename Async_Read_Stream, typename Mutable_Buffers,
    typename Handler>
inline void async_read(Async_Read_Stream& s, const Mutable_Buffers& buffers,
    Handler handler)
{
  async_read(s, buffers, transfer_all(), handler);
}

namespace detail
{
  template <typename Async_Read_Stream, typename Allocator,
      typename Completion_Condition, typename Handler>
  class read_streambuf_handler
  {
  public:
    read_streambuf_handler(Async_Read_Stream& stream,
        basic_streambuf<Allocator>& streambuf,
        Completion_Condition completion_condition, Handler handler)
      : stream_(stream),
        streambuf_(streambuf),
        total_transferred_(0),
        completion_condition_(completion_condition),
        handler_(handler)
    {
    }

    void operator()(const typename Async_Read_Stream::error_type& e,
        std::size_t bytes_transferred)
    {
      total_transferred_ += bytes_transferred;
      streambuf_.commit(bytes_transferred);
      if (completion_condition_(e, total_transferred_))
      {
        stream_.io_service().dispatch(
            detail::bind_handler(handler_, e, total_transferred_));
      }
      else
      {
        stream_.async_read_some(streambuf_.prepare(512), *this);
      }
    }

    friend void* asio_handler_allocate(std::size_t size,
        read_streambuf_handler<Async_Read_Stream, Allocator,
          Completion_Condition, Handler>* this_handler)
    {
      return asio_handler_alloc_helpers::allocate(
          size, &this_handler->handler_);
    }

    friend void asio_handler_deallocate(void* pointer, std::size_t size,
        read_streambuf_handler<Async_Read_Stream, Allocator,
          Completion_Condition, Handler>* this_handler)
    {
      asio_handler_alloc_helpers::deallocate(
          pointer, size, &this_handler->handler_);
    }

  private:
    Async_Read_Stream& stream_;
    asio::basic_streambuf<Allocator>& streambuf_;
    std::size_t total_transferred_;
    Completion_Condition completion_condition_;
    Handler handler_;
  };
} // namespace detail

template <typename Async_Read_Stream, typename Allocator,
    typename Completion_Condition, typename Handler>
inline void async_read(Async_Read_Stream& s,
    asio::basic_streambuf<Allocator>& b,
    Completion_Condition completion_condition, Handler handler)
{
  s.async_read_some(b.prepare(512),
      detail::read_streambuf_handler<Async_Read_Stream, Allocator,
        Completion_Condition, Handler>(
          s, b, completion_condition, handler));
}

template <typename Async_Read_Stream, typename Allocator, typename Handler>
inline void async_read(Async_Read_Stream& s,
    asio::basic_streambuf<Allocator>& b, Handler handler)
{
  async_read(s, b, transfer_all(), handler);
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_READ_IPP
