//
// write.ipp
// ~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WRITE_IPP
#define ASIO_WRITE_IPP

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

template <typename Sync_Write_Stream, typename Const_Buffers,
    typename Completion_Condition, typename Error_Handler>
std::size_t write(Sync_Write_Stream& s, const Const_Buffers& buffers,
    Completion_Condition completion_condition, Error_Handler error_handler)
{
  asio::detail::consuming_buffers<
    const_buffer, Const_Buffers> tmp(buffers);
  std::size_t total_transferred = 0;
  while (tmp.begin() != tmp.end())
  {
    typename Sync_Write_Stream::error_type e;
    std::size_t bytes_transferred = s.write_some(tmp, assign_error(e));
    tmp.consume(bytes_transferred);
    total_transferred += bytes_transferred;
    if (completion_condition(e, total_transferred))
    {
      error_handler(e);
      return total_transferred;
    }
  }
  typename Sync_Write_Stream::error_type e;
  error_handler(e);
  return total_transferred;
}

template <typename Sync_Write_Stream, typename Const_Buffers>
inline std::size_t write(Sync_Write_Stream& s, const Const_Buffers& buffers)
{
  return write(s, buffers, transfer_all(), throw_error());
}

template <typename Sync_Write_Stream, typename Const_Buffers,
    typename Completion_Condition>
inline std::size_t write(Sync_Write_Stream& s, const Const_Buffers& buffers,
    Completion_Condition completion_condition)
{
  return write(s, buffers, completion_condition, throw_error());
}

template <typename Sync_Write_Stream, typename Allocator,
    typename Completion_Condition, typename Error_Handler>
std::size_t write(Sync_Write_Stream& s,
    asio::basic_streambuf<Allocator>& b,
    Completion_Condition completion_condition, Error_Handler error_handler)
{
  typename Sync_Write_Stream::error_type error;
  std::size_t bytes_transferred = write(s, b.data(),
      completion_condition, asio::assign_error(error));
  b.consume(bytes_transferred);
  error_handler(error);
  return bytes_transferred;
}

template <typename Sync_Write_Stream, typename Allocator>
inline std::size_t write(Sync_Write_Stream& s,
    asio::basic_streambuf<Allocator>& b)
{
  return write(s, b, transfer_all(), throw_error());
}

template <typename Sync_Write_Stream, typename Allocator,
    typename Completion_Condition>
inline std::size_t write(Sync_Write_Stream& s,
    asio::basic_streambuf<Allocator>& b,
    Completion_Condition completion_condition)
{
  return write(s, b, completion_condition, throw_error());
}

namespace detail
{
  template <typename Async_Write_Stream, typename Const_Buffers,
      typename Completion_Condition, typename Handler>
  class write_handler
  {
  public:
    write_handler(Async_Write_Stream& stream, const Const_Buffers& buffers,
        Completion_Condition completion_condition, Handler handler)
      : stream_(stream),
        buffers_(buffers),
        total_transferred_(0),
        completion_condition_(completion_condition),
        handler_(handler)
    {
    }

    void operator()(const typename Async_Write_Stream::error_type& e,
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
        stream_.async_write_some(buffers_, *this);
      }
    }

    friend void* asio_handler_allocate(std::size_t size,
        write_handler<Async_Write_Stream, Const_Buffers,
          Completion_Condition, Handler>* this_handler)
    {
      return asio_handler_alloc_helpers::allocate(
          size, &this_handler->handler_);
    }

    friend void asio_handler_deallocate(void* pointer, std::size_t size,
        write_handler<Async_Write_Stream, Const_Buffers,
          Completion_Condition, Handler>* this_handler)
    {
      asio_handler_alloc_helpers::deallocate(
          pointer, size, &this_handler->handler_);
    }

  private:
    Async_Write_Stream& stream_;
    asio::detail::consuming_buffers<
      const_buffer, Const_Buffers> buffers_;
    std::size_t total_transferred_;
    Completion_Condition completion_condition_;
    Handler handler_;
  };
} // namespace detail

template <typename Async_Write_Stream, typename Const_Buffers,
  typename Completion_Condition, typename Handler>
inline void async_write(Async_Write_Stream& s, const Const_Buffers& buffers,
    Completion_Condition completion_condition, Handler handler)
{
  s.async_write_some(buffers,
      detail::write_handler<Async_Write_Stream, Const_Buffers,
        Completion_Condition, Handler>(
          s, buffers, completion_condition, handler));
}

template <typename Async_Write_Stream, typename Const_Buffers, typename Handler>
inline void async_write(Async_Write_Stream& s, const Const_Buffers& buffers,
    Handler handler)
{
  async_write(s, buffers, transfer_all(), handler);
}

namespace detail
{
  template <typename Async_Write_Stream, typename Allocator, typename Handler>
  class write_streambuf_handler
  {
  public:
    write_streambuf_handler(asio::basic_streambuf<Allocator>& streambuf,
        Handler handler)
      : streambuf_(streambuf),
        handler_(handler)
    {
    }

    void operator()(const typename Async_Write_Stream::error_type& e,
        std::size_t bytes_transferred)
    {
      streambuf_.consume(bytes_transferred);
      handler_(e, bytes_transferred);
    }

    friend void* asio_handler_allocate(std::size_t size,
        write_streambuf_handler<Async_Write_Stream,
          Allocator, Handler>* this_handler)
    {
      return asio_handler_alloc_helpers::allocate(
          size, &this_handler->handler_);
    }

    friend void asio_handler_deallocate(void* pointer, std::size_t size,
        write_streambuf_handler<Async_Write_Stream,
          Allocator, Handler>* this_handler)
    {
      asio_handler_alloc_helpers::deallocate(
          pointer, size, &this_handler->handler_);
    }

  private:
    asio::basic_streambuf<Allocator>& streambuf_;
    Handler handler_;
  };
} // namespace detail

template <typename Async_Write_Stream, typename Allocator,
  typename Completion_Condition, typename Handler>
inline void async_write(Async_Write_Stream& s,
    asio::basic_streambuf<Allocator>& b,
    Completion_Condition completion_condition, Handler handler)
{
  async_write(s, b.data(),
      detail::write_streambuf_handler<Async_Write_Stream, Allocator, Handler>(
        b, handler));
}

template <typename Async_Write_Stream, typename Allocator, typename Handler>
inline void async_write(Async_Write_Stream& s,
    asio::basic_streambuf<Allocator>& b, Handler handler)
{
  async_write(s, b, transfer_all(), handler);
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_WRITE_IPP
