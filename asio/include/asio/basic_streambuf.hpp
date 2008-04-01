//
// basic_streambuf.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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

#include "asio/buffer.hpp"
#include "asio/detail/noncopyable.hpp"

namespace asio {

/// Automatically resizable buffer class based on std::streambuf.
template <typename Allocator = std::allocator<char> >
class basic_streambuf
  : public std::streambuf,
    private noncopyable
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The type used to represent the get area as a list of buffers.
  typedef implementation_defined const_buffers_type;

  /// The type used to represent the put area as a list of buffers.
  typedef implementation_defined mutable_buffers_type;
#else
  typedef asio::const_buffers_1 const_buffers_type;
  typedef asio::mutable_buffers_1 mutable_buffers_type;
#endif

  /// Construct a buffer with a specified maximum size.
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

  /// Return the size of the get area in characters.
  std::size_t size() const
  {
    return pptr() - gptr();
  }

  /// Return the maximum size of the buffer.
  std::size_t max_size() const
  {
    return max_size_;
  }

  /// Get a list of buffers that represents the get area.
  const_buffers_type data() const
  {
    return asio::buffer(asio::const_buffer(gptr(),
          (pptr() - gptr()) * sizeof(char_type)));
  }

  /// Get a list of buffers that represents the put area, with the given size.
  mutable_buffers_type prepare(std::size_t size)
  {
    reserve(size);
    return asio::buffer(asio::mutable_buffer(
          pptr(), size * sizeof(char_type)));
  }

  /// Move the start of the put area by the specified number of characters.
  void commit(std::size_t n)
  {
    if (pptr() + n > epptr())
      n = epptr() - pptr();
    pbump(static_cast<int>(n));
  }

  /// Move the start of the get area by the specified number of characters.
  void consume(std::size_t n)
  {
    if (gptr() + n > pptr())
      n = pptr() - gptr();
    gbump(static_cast<int>(n));
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
      if (n <= max_size_ && pnext <= max_size_ - n)
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

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_STREAMBUF_HPP
