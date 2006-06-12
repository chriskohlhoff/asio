//
// basic_streambuf.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_STREAMBUF_HPP
#define ASIO_BASIC_STREAMBUF_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <streambuf>
#include <vector>
#include "asio/detail/pop_options.hpp"

#include "asio/completion_condition.hpp"
#include "asio/error_handler.hpp"
#include "asio/write.hpp"
#include "asio/detail/noncopyable.hpp"

namespace asio {

template <typename Allocator = std::allocator<char> >
class basic_streambuf
  : public std::streambuf,
    private noncopyable
{
public:
  typedef asio::const_buffer_container_1 const_buffers_type;
  typedef asio::mutable_buffer_container_1 mutable_buffers_type;

  explicit basic_streambuf(
      std::size_t max_size = (std::numeric_limits<std::size_t>::max)(),
      const Allocator& allocator = Allocator())
    : max_size_(max_size),
      buffer_(allocator)
  {
    std::size_t pend = (std::min<std::size_t>)(max_size_, buffer_delta);
    buffer_.resize((std::max<std::size_t>)(pend, 1));
    setg(&buffer_[0], &buffer_[0], &buffer_[0]);
    setp(&buffer_[0], &buffer_[0] + pend);
  }

  void sbump(std::streamsize n)
  {
    while (n > 0)
    {
      sbumpc();
      --n;
    }
  }

  void spbump(std::streamsize n)
  {
    if (pptr() + n > epptr())
      n = epptr() - pptr();
    pbump(n);
  }

  int_type spbumpc()
  {
    if (pptr() == epptr())
    {
      return traits_type::eof();
    }

    int_type c = traits_type::to_int_type(*pptr());
    pbump(1);
    return c;
  }

  const_buffers_type sbuffers() const
  {
    return asio::buffer(asio::const_buffer(gptr(),
          (pptr() - gptr()) * sizeof(char_type)));
  }

  mutable_buffers_type spbuffers(std::size_t size)
  {
    reserve(size);
    return asio::buffer(asio::mutable_buffer(
          pptr(), size * sizeof(char_type)));
  }

protected:
  enum { buffer_delta = 128 };

  int_type underflow()
  {
    if (gptr() < pptr())
    {
      setg(&buffer_[0], gptr(), pptr());
      return traits_type::to_int_type(*gptr());
    }
    else
    {
      return traits_type::eof();
    }
  }

  int_type overflow(int_type c)
  {
    if (!traits_type::eq_int_type(c, traits_type::eof()))
    {
      if (pptr() == epptr())
      {
        std::size_t buffer_size = pptr() - gptr();
        if (buffer_size < max_size_ && max_size_ - buffer_size < buffer_delta)
        {
          reserve(max_size_ - buffer_size);
        }
        else
        {
          reserve(buffer_delta);
        }
      }

      *pptr() = traits_type::to_char_type(c);
      pbump(1);
      return c;
    }

    return traits_type::not_eof(c);
  }

  void reserve(std::size_t n)
  {
    // Get current stream positions as offsets.
    std::size_t gnext = gptr() - &buffer_[0];
    std::size_t gend = egptr() - &buffer_[0];
    std::size_t pnext = pptr() - &buffer_[0];
    std::size_t pend = epptr() - &buffer_[0];

    // Check if there is already enough space in the put area.
    if (n <= pend - pnext)
    {
      return;
    }

    // Shift existing contents of get area to start of buffer.
    if (gnext > 0)
    {
      std::rotate(&buffer_[0], &buffer_[0] + gnext, &buffer_[0] + pend);
      gend -= gnext;
      pnext -= gnext;
    }

    // Ensure buffer is large enough to hold at least the specified size.
    if (n > pend - pnext)
    {
      if (n <= max_size_ & pnext <= max_size_ - n)
      {
        buffer_.resize((std::max<std::size_t>)(pnext + n, 1));
      }
      else
      {
        throw std::length_error("asio::streambuf too long");
      }
    }

    // Update stream positions.
    setg(&buffer_[0], &buffer_[0], &buffer_[0] + gend);
    setp(&buffer_[0] + pnext, &buffer_[0] + pnext + n);
  }

private:
  std::size_t max_size_;
  std::vector<char_type, Allocator> buffer_;
};

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
        b.spbuffers(512), assign_error(e));
    b.spbump(bytes_transferred);
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
      streambuf_.spbump(bytes_transferred);
      if (completion_condition_(e, total_transferred_))
      {
        stream_.io_service().dispatch(
            detail::bind_handler(handler_, e, total_transferred_));
      }
      else
      {
        stream_.async_read_some(streambuf_.spbuffers(512), *this);
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
  s.async_read_some(b.spbuffers(512),
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

template <typename Sync_Write_Stream, typename Allocator,
    typename Completion_Condition, typename Error_Handler>
std::size_t write(Sync_Write_Stream& s,
    asio::basic_streambuf<Allocator>& b,
    Completion_Condition completion_condition, Error_Handler error_handler)
{
  typename Sync_Write_Stream::error_type error;
  std::size_t bytes_transferred = write(s, b.sbuffers(),
      completion_condition, asio::assign_error(error));
  b.sbump(bytes_transferred);
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
      streambuf_.sbump(bytes_transferred);
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
  async_write(s, b.sbuffers(),
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

#endif // ASIO_BASIC_STREAMBUF_HPP
