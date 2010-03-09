//
// read.ipp
// ~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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

#include "asio/detail/push_options.hpp"
#include <algorithm>
#include "asio/detail/pop_options.hpp"

#include "asio/buffer.hpp"
#include "asio/completion_condition.hpp"
#include "asio/error.hpp"
#include "asio/detail/base_from_completion_cond.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/consuming_buffers.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/throw_error.hpp"

namespace asio {

template <typename SyncReadStream, typename MutableBufferSequence,
    typename CompletionCondition>
std::size_t read(SyncReadStream& s, const MutableBufferSequence& buffers,
    CompletionCondition completion_condition, asio::error_code& ec)
{
  ec = asio::error_code();
  asio::detail::consuming_buffers<
    mutable_buffer, MutableBufferSequence> tmp(buffers);
  std::size_t total_transferred = 0;
  tmp.prepare(detail::adapt_completion_condition_result(
        completion_condition(ec, total_transferred)));
  while (tmp.begin() != tmp.end())
  {
    std::size_t bytes_transferred = s.read_some(tmp, ec);
    tmp.consume(bytes_transferred);
    total_transferred += bytes_transferred;
    tmp.prepare(detail::adapt_completion_condition_result(
          completion_condition(ec, total_transferred)));
  }
  return total_transferred;
}

template <typename SyncReadStream, typename MutableBufferSequence>
inline std::size_t read(SyncReadStream& s, const MutableBufferSequence& buffers)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read(s, buffers, transfer_all(), ec);
  asio::detail::throw_error(ec);
  return bytes_transferred;
}

template <typename SyncReadStream, typename MutableBufferSequence,
    typename CompletionCondition>
inline std::size_t read(SyncReadStream& s, const MutableBufferSequence& buffers,
    CompletionCondition completion_condition)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read(s, buffers, completion_condition, ec);
  asio::detail::throw_error(ec);
  return bytes_transferred;
}

#if !defined(BOOST_NO_IOSTREAM)

template <typename SyncReadStream, typename Allocator,
    typename CompletionCondition>
std::size_t read(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition, asio::error_code& ec)
{
  ec = asio::error_code();
  std::size_t total_transferred = 0;
  std::size_t max_size = detail::adapt_completion_condition_result(
        completion_condition(ec, total_transferred));
  std::size_t bytes_available = std::min<std::size_t>(512,
      std::min<std::size_t>(max_size, b.max_size() - b.size()));
  while (bytes_available > 0)
  {
    std::size_t bytes_transferred = s.read_some(b.prepare(bytes_available), ec);
    b.commit(bytes_transferred);
    total_transferred += bytes_transferred;
    max_size = detail::adapt_completion_condition_result(
          completion_condition(ec, total_transferred));
    bytes_available = std::min<std::size_t>(512,
        std::min<std::size_t>(max_size, b.max_size() - b.size()));
  }
  return total_transferred;
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read(s, b, transfer_all(), ec);
  asio::detail::throw_error(ec);
  return bytes_transferred;
}

template <typename SyncReadStream, typename Allocator,
    typename CompletionCondition>
inline std::size_t read(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read(s, b, completion_condition, ec);
  asio::detail::throw_error(ec);
  return bytes_transferred;
}

#endif // !defined(BOOST_NO_IOSTREAM)

namespace detail
{
  template <typename AsyncReadStream, typename MutableBufferSequence,
      typename CompletionCondition, typename ReadHandler>
  class read_op
    : detail::base_from_completion_cond<CompletionCondition>
  {
  public:
    read_op(AsyncReadStream& stream, const MutableBufferSequence& buffers,
        CompletionCondition completion_condition, ReadHandler handler)
      : detail::base_from_completion_cond<
          CompletionCondition>(completion_condition),
        stream_(stream),
        buffers_(buffers),
        total_transferred_(0),
        handler_(handler),
        start_(true)
    {
    }

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred)
    {
      switch (start_)
      {
        case true: start_ = false;
        buffers_.prepare(this->check(ec, total_transferred_));
        for (;;)
        {
          stream_.async_read_some(buffers_, *this);
          return; default:
          total_transferred_ += bytes_transferred;
          buffers_.consume(bytes_transferred);
          buffers_.prepare(this->check(ec, total_transferred_));
          if ((!ec && bytes_transferred == 0)
              || buffers_.begin() == buffers_.end())
            break;
        }

        handler_(ec, total_transferred_);
      }
    }

  //private:
    AsyncReadStream& stream_;
    asio::detail::consuming_buffers<
      mutable_buffer, MutableBufferSequence> buffers_;
    std::size_t total_transferred_;
    ReadHandler handler_;
    bool start_;
  };

  template <typename AsyncReadStream,
      typename CompletionCondition, typename ReadHandler>
  class read_op<AsyncReadStream, asio::mutable_buffers_1,
      CompletionCondition, ReadHandler>
    : detail::base_from_completion_cond<CompletionCondition>
  {
  public:
    read_op(AsyncReadStream& stream,
        const asio::mutable_buffers_1& buffers,
        CompletionCondition completion_condition,
        ReadHandler handler)
      : detail::base_from_completion_cond<
          CompletionCondition>(completion_condition),
        stream_(stream),
        buffer_(buffers),
        total_transferred_(0),
        handler_(handler),
        start_(true)
    {
    }

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred)
    {
      std::size_t n = 0;
      switch (start_)
      {
        case true: start_ = false;
        n = this->check(ec, total_transferred_);
        for (;;)
        {
          stream_.async_read_some(asio::buffer(
                buffer_ + total_transferred_, n), *this);
          return; default:
          total_transferred_ += bytes_transferred;
          if ((!ec && bytes_transferred == 0)
              || (n = this->check(ec, total_transferred_)) == 0
              || total_transferred_ == asio::buffer_size(buffer_))
            break;
        }

        handler_(ec, total_transferred_);
      }
    }

  //private:
    AsyncReadStream& stream_;
    asio::mutable_buffer buffer_;
    std::size_t total_transferred_;
    ReadHandler handler_;
    bool start_;
  };

  template <typename AsyncReadStream, typename MutableBufferSequence,
      typename CompletionCondition, typename ReadHandler>
  inline void* asio_handler_allocate(std::size_t size,
      read_op<AsyncReadStream, MutableBufferSequence,
        CompletionCondition, ReadHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename MutableBufferSequence,
      typename CompletionCondition, typename ReadHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      read_op<AsyncReadStream, MutableBufferSequence,
        CompletionCondition, ReadHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename MutableBufferSequence, typename CompletionCondition,
      typename ReadHandler>
  inline void asio_handler_invoke(const Function& function,
      read_op<AsyncReadStream, MutableBufferSequence,
        CompletionCondition, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }
} // namespace detail

template <typename AsyncReadStream, typename MutableBufferSequence,
    typename CompletionCondition, typename ReadHandler>
inline void async_read(AsyncReadStream& s, const MutableBufferSequence& buffers,
    CompletionCondition completion_condition, ReadHandler handler)
{
  detail::read_op<AsyncReadStream, MutableBufferSequence,
    CompletionCondition, ReadHandler>(
      s, buffers, completion_condition, handler)(
        asio::error_code(), 0);
}

template <typename AsyncReadStream, typename MutableBufferSequence,
    typename ReadHandler>
inline void async_read(AsyncReadStream& s, const MutableBufferSequence& buffers,
    ReadHandler handler)
{
  async_read(s, buffers, transfer_all(), handler);
}

#if !defined(BOOST_NO_IOSTREAM)

namespace detail
{
  template <typename AsyncReadStream, typename Allocator,
      typename CompletionCondition, typename ReadHandler>
  class read_streambuf_op
    : detail::base_from_completion_cond<CompletionCondition>
  {
  public:
    read_streambuf_op(AsyncReadStream& stream,
        basic_streambuf<Allocator>& streambuf,
        CompletionCondition completion_condition, ReadHandler handler)
      : detail::base_from_completion_cond<
          CompletionCondition>(completion_condition),
        stream_(stream),
        streambuf_(streambuf),
        total_transferred_(0),
        handler_(handler),
        start_(true)
    {
    }

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred)
    {
      std::size_t max_size, bytes_available;
      switch (start_)
      {
        case true: start_ = false;
        max_size = this->check(ec, total_transferred_);
        bytes_available = std::min<std::size_t>(512,
            std::min<std::size_t>(max_size,
              streambuf_.max_size() - streambuf_.size()));
        for (;;)
        {
          stream_.async_read_some(streambuf_.prepare(bytes_available), *this);
          return; default:
          total_transferred_ += bytes_transferred;
          streambuf_.commit(bytes_transferred);
          max_size = this->check(ec, total_transferred_);
          bytes_available = std::min<std::size_t>(512,
              std::min<std::size_t>(max_size,
                streambuf_.max_size() - streambuf_.size()));
          if ((!ec && bytes_transferred == 0) || bytes_available == 0)
            break;
        }

        handler_(ec, total_transferred_);
      }
    }

  //private:
    AsyncReadStream& stream_;
    asio::basic_streambuf<Allocator>& streambuf_;
    std::size_t total_transferred_;
    ReadHandler handler_;
    bool start_;
  };

  template <typename AsyncReadStream, typename Allocator,
      typename CompletionCondition, typename ReadHandler>
  inline void* asio_handler_allocate(std::size_t size,
      read_streambuf_op<AsyncReadStream, Allocator,
        CompletionCondition, ReadHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename Allocator,
      typename CompletionCondition, typename ReadHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      read_streambuf_op<AsyncReadStream, Allocator,
        CompletionCondition, ReadHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename Allocator, typename CompletionCondition, typename ReadHandler>
  inline void asio_handler_invoke(const Function& function,
      read_streambuf_op<AsyncReadStream, Allocator,
        CompletionCondition, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }
} // namespace detail

template <typename AsyncReadStream, typename Allocator,
    typename CompletionCondition, typename ReadHandler>
inline void async_read(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition, ReadHandler handler)
{
  detail::read_streambuf_op<AsyncReadStream,
    Allocator, CompletionCondition, ReadHandler>(
      s, b, completion_condition, handler)(
        asio::error_code(), 0);
}

template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
inline void async_read(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, ReadHandler handler)
{
  async_read(s, b, transfer_all(), handler);
}

#endif // !defined(BOOST_NO_IOSTREAM)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_READ_IPP
