//
// write_at.ipp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WRITE_AT_IPP
#define ASIO_WRITE_AT_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/buffer.hpp"
#include "asio/completion_condition.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/consuming_buffers.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/throw_error.hpp"

namespace asio {

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence,
    typename CompletionCondition>
std::size_t write_at(SyncRandomAccessWriteDevice& d,
    boost::uint64_t offset, const ConstBufferSequence& buffers,
    CompletionCondition completion_condition, asio::error_code& ec)
{
  asio::detail::consuming_buffers<
    const_buffer, ConstBufferSequence> tmp(buffers);
  std::size_t total_transferred = 0;
  while (tmp.begin() != tmp.end())
  {
    std::size_t bytes_transferred = d.write_some_at(
        offset + total_transferred, tmp, ec);
    tmp.consume(bytes_transferred);
    total_transferred += bytes_transferred;
    if (completion_condition(ec, total_transferred))
      return total_transferred;
  }
  ec = asio::error_code();
  return total_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
    boost::uint64_t offset, const ConstBufferSequence& buffers)
{
  asio::error_code ec;
  std::size_t bytes_transferred = write_at(
      d, offset, buffers, transfer_all(), ec);
  asio::detail::throw_error(ec);
  return bytes_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence,
    typename CompletionCondition>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
    boost::uint64_t offset, const ConstBufferSequence& buffers,
    CompletionCondition completion_condition)
{
  asio::error_code ec;
  std::size_t bytes_transferred = write_at(
      d, offset, buffers, completion_condition, ec);
  asio::detail::throw_error(ec);
  return bytes_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename Allocator,
    typename CompletionCondition>
std::size_t write_at(SyncRandomAccessWriteDevice& d,
    boost::uint64_t offset, asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition, asio::error_code& ec)
{
  std::size_t bytes_transferred = write_at(
      d, offset, b.data(), completion_condition, ec);
  b.consume(bytes_transferred);
  return bytes_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename Allocator>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
    boost::uint64_t offset, asio::basic_streambuf<Allocator>& b)
{
  asio::error_code ec;
  std::size_t bytes_transferred = write_at(d, offset, b, transfer_all(), ec);
  asio::detail::throw_error(ec);
  return bytes_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename Allocator,
    typename CompletionCondition>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
    boost::uint64_t offset, asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition)
{
  asio::error_code ec;
  std::size_t bytes_transferred = write_at(
      d, offset, b, completion_condition, ec);
  asio::detail::throw_error(ec);
  return bytes_transferred;
}

namespace detail
{
  template <typename AsyncRandomAccessWriteDevice, typename ConstBufferSequence,
      typename CompletionCondition, typename WriteHandler>
  class write_at_handler
  {
  public:
    typedef asio::detail::consuming_buffers<
      const_buffer, ConstBufferSequence> buffers_type;

    write_at_handler(AsyncRandomAccessWriteDevice& stream,
        boost::uint64_t offset, const buffers_type& buffers,
        CompletionCondition completion_condition, WriteHandler handler)
      : stream_(stream),
        buffers_(buffers),
        offset_(offset),
        total_transferred_(0),
        completion_condition_(completion_condition),
        handler_(handler)
    {
    }

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred)
    {
      total_transferred_ += bytes_transferred;
      buffers_.consume(bytes_transferred);
      if (completion_condition_(ec, total_transferred_)
          || buffers_.begin() == buffers_.end())
      {
        handler_(ec, total_transferred_);
      }
      else
      {
        stream_.async_write_some_at(
            offset_ + total_transferred_, buffers_, *this);
      }
    }

  //private:
    AsyncRandomAccessWriteDevice& stream_;
    buffers_type buffers_;
    boost::uint64_t offset_;
    std::size_t total_transferred_;
    CompletionCondition completion_condition_;
    WriteHandler handler_;
  };

  template <typename AsyncRandomAccessWriteDevice, typename ConstBufferSequence,
      typename CompletionCondition, typename WriteHandler>
  inline void* asio_handler_allocate(std::size_t size,
      write_at_handler<AsyncRandomAccessWriteDevice, ConstBufferSequence,
        CompletionCondition, WriteHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, &this_handler->handler_);
  }

  template <typename AsyncRandomAccessWriteDevice, typename ConstBufferSequence,
      typename CompletionCondition, typename WriteHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      write_at_handler<AsyncRandomAccessWriteDevice, ConstBufferSequence,
        CompletionCondition, WriteHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, &this_handler->handler_);
  }

  template <typename Function, typename AsyncRandomAccessWriteDevice,
      typename ConstBufferSequence, typename CompletionCondition,
      typename WriteHandler>
  inline void asio_handler_invoke(const Function& function,
      write_at_handler<AsyncRandomAccessWriteDevice, ConstBufferSequence,
        CompletionCondition, WriteHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, &this_handler->handler_);
  }
} // namespace detail

template <typename AsyncRandomAccessWriteDevice, typename ConstBufferSequence,
    typename CompletionCondition, typename WriteHandler>
inline void async_write_at(AsyncRandomAccessWriteDevice& d,
    boost::uint64_t offset, const ConstBufferSequence& buffers,
    CompletionCondition completion_condition, WriteHandler handler)
{
  asio::detail::consuming_buffers<
    const_buffer, ConstBufferSequence> tmp(buffers);
  d.async_write_some_at(offset, tmp,
      detail::write_at_handler<AsyncRandomAccessWriteDevice,
      ConstBufferSequence, CompletionCondition, WriteHandler>(
          d, offset, tmp, completion_condition, handler));
}

template <typename AsyncRandomAccessWriteDevice, typename ConstBufferSequence,
    typename WriteHandler>
inline void async_write_at(AsyncRandomAccessWriteDevice& d,
    boost::uint64_t offset, const ConstBufferSequence& buffers,
    WriteHandler handler)
{
  async_write_at(d, offset, buffers, transfer_all(), handler);
}

namespace detail
{
  template <typename AsyncRandomAccessWriteDevice, typename Allocator,
      typename WriteHandler>
  class write_at_streambuf_handler
  {
  public:
    write_at_streambuf_handler(
        asio::basic_streambuf<Allocator>& streambuf,
        WriteHandler handler)
      : streambuf_(streambuf),
        handler_(handler)
    {
    }

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred)
    {
      streambuf_.consume(bytes_transferred);
      handler_(ec, bytes_transferred);
    }

  //private:
    asio::basic_streambuf<Allocator>& streambuf_;
    WriteHandler handler_;
  };

  template <typename AsyncRandomAccessWriteDevice, typename Allocator,
      typename WriteHandler>
  inline void* asio_handler_allocate(std::size_t size,
      write_at_streambuf_handler<AsyncRandomAccessWriteDevice,
        Allocator, WriteHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, &this_handler->handler_);
  }

  template <typename AsyncRandomAccessWriteDevice, typename Allocator,
      typename WriteHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      write_at_streambuf_handler<AsyncRandomAccessWriteDevice,
        Allocator, WriteHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, &this_handler->handler_);
  }

  template <typename Function, typename AsyncRandomAccessWriteDevice,
      typename Allocator, typename WriteHandler>
  inline void asio_handler_invoke(const Function& function,
      write_at_streambuf_handler<AsyncRandomAccessWriteDevice,
        Allocator, WriteHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, &this_handler->handler_);
  }
} // namespace detail

template <typename AsyncRandomAccessWriteDevice, typename Allocator,
    typename CompletionCondition, typename WriteHandler>
inline void async_write_at(AsyncRandomAccessWriteDevice& d,
    boost::uint64_t offset, asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition, WriteHandler handler)
{
  async_write_at(d, offset, b.data(), completion_condition,
      detail::write_at_streambuf_handler<
        AsyncRandomAccessWriteDevice, Allocator, WriteHandler>(b, handler));
}

template <typename AsyncRandomAccessWriteDevice, typename Allocator,
    typename WriteHandler>
inline void async_write_at(AsyncRandomAccessWriteDevice& d,
    boost::uint64_t offset, asio::basic_streambuf<Allocator>& b,
    WriteHandler handler)
{
  async_write_at(d, offset, b, transfer_all(), handler);
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_WRITE_AT_IPP
