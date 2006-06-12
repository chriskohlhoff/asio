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

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_STREAMBUF_HPP
