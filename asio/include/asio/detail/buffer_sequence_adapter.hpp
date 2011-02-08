//
// detail/buffer_sequence_adapter.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_BUFFER_SEQUENCE_ADAPTER_HPP
#define ASIO_DETAIL_BUFFER_SEQUENCE_ADAPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/buffer.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class buffer_sequence_adapter_base
{
protected:
#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
  typedef WSABUF native_buffer_type;

  static void init_native_buffer(WSABUF& buf,
      const asio::mutable_buffer& buffer)
  {
    buf.buf = asio::buffer_cast<char*>(buffer);
    buf.len = static_cast<ULONG>(asio::buffer_size(buffer));
  }

  static void init_native_buffer(WSABUF& buf,
      const asio::const_buffer& buffer)
  {
    buf.buf = const_cast<char*>(asio::buffer_cast<const char*>(buffer));
    buf.len = static_cast<ULONG>(asio::buffer_size(buffer));
  }
#else // defined(BOOST_WINDOWS) || defined(__CYGWIN__)
  typedef iovec native_buffer_type;

  static void init_iov_base(void*& base, void* addr)
  {
    base = addr;
  }

  template <typename T>
  static void init_iov_base(T& base, void* addr)
  {
    base = static_cast<T>(addr);
  }

  static void init_native_buffer(iovec& iov,
      const asio::mutable_buffer& buffer)
  {
    init_iov_base(iov.iov_base, asio::buffer_cast<void*>(buffer));
    iov.iov_len = asio::buffer_size(buffer);
  }

  static void init_native_buffer(iovec& iov,
      const asio::const_buffer& buffer)
  {
    init_iov_base(iov.iov_base, const_cast<void*>(
          asio::buffer_cast<const void*>(buffer)));
    iov.iov_len = asio::buffer_size(buffer);
  }
#endif // defined(BOOST_WINDOWS) || defined(__CYGWIN__)
};

// Helper class to translate buffers into the native buffer representation.
template <typename Buffer, typename Buffers>
class buffer_sequence_adapter
  : buffer_sequence_adapter_base
{
public:
  explicit buffer_sequence_adapter(const Buffers& buffers)
    : count_(0), total_buffer_size_(0)
  {
    typename Buffers::const_iterator iter = buffers.begin();
    typename Buffers::const_iterator end = buffers.end();
    for (; iter != end && count_ < max_buffers; ++iter, ++count_)
    {
      Buffer buffer(*iter);
      init_native_buffer(buffers_[count_], buffer);
      total_buffer_size_ += asio::buffer_size(buffer);
    }
  }

  native_buffer_type* buffers()
  {
    return buffers_;
  }

  std::size_t count() const
  {
    return count_;
  }

  bool all_empty() const
  {
    return total_buffer_size_ == 0;
  }

  static bool all_empty(const Buffers& buffers)
  {
    typename Buffers::const_iterator iter = buffers.begin();
    typename Buffers::const_iterator end = buffers.end();
    std::size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
      if (asio::buffer_size(Buffer(*iter)) > 0)
        return false;
    return true;
  }

  static void validate(const Buffers& buffers)
  {
    typename Buffers::const_iterator iter = buffers.begin();
    typename Buffers::const_iterator end = buffers.end();
    for (; iter != end; ++iter)
    {
      Buffer buffer(*iter);
      asio::buffer_cast<const void*>(buffer);
    }
  }

  static Buffer first(const Buffers& buffers)
  {
    typename Buffers::const_iterator iter = buffers.begin();
    typename Buffers::const_iterator end = buffers.end();
    for (; iter != end; ++iter)
    {
      Buffer buffer(*iter);
      if (asio::buffer_size(buffer) != 0)
        return buffer;
    }
    return Buffer();
  }

private:
  // The maximum number of buffers to support in a single operation.
  enum { max_buffers = 64 < max_iov_len ? 64 : max_iov_len };

  native_buffer_type buffers_[max_buffers];
  std::size_t count_;
  std::size_t total_buffer_size_;
};

template <typename Buffer>
class buffer_sequence_adapter<Buffer, asio::mutable_buffers_1>
  : buffer_sequence_adapter_base
{
public:
  explicit buffer_sequence_adapter(
      const asio::mutable_buffers_1& buffers)
  {
    init_native_buffer(buffer_, Buffer(buffers));
    total_buffer_size_ = asio::buffer_size(buffers);
  }

  native_buffer_type* buffers()
  {
    return &buffer_;
  }

  std::size_t count() const
  {
    return 1;
  }

  bool all_empty() const
  {
    return total_buffer_size_ == 0;
  }

  static bool all_empty(const asio::mutable_buffers_1& buffers)
  {
    return asio::buffer_size(buffers) == 0;
  }

  static void validate(const asio::mutable_buffers_1& buffers)
  {
    asio::buffer_cast<const void*>(buffers);
  }

  static Buffer first(const asio::mutable_buffers_1& buffers)
  {
    return Buffer(buffers);
  }

private:
  native_buffer_type buffer_;
  std::size_t total_buffer_size_;
};

template <typename Buffer>
class buffer_sequence_adapter<Buffer, asio::const_buffers_1>
  : buffer_sequence_adapter_base
{
public:
  explicit buffer_sequence_adapter(
      const asio::const_buffers_1& buffers)
  {
    init_native_buffer(buffer_, Buffer(buffers));
    total_buffer_size_ = asio::buffer_size(buffers);
  }

  native_buffer_type* buffers()
  {
    return &buffer_;
  }

  std::size_t count() const
  {
    return 1;
  }

  bool all_empty() const
  {
    return total_buffer_size_ == 0;
  }

  static bool all_empty(const asio::const_buffers_1& buffers)
  {
    return asio::buffer_size(buffers) == 0;
  }

  static void validate(const asio::const_buffers_1& buffers)
  {
    asio::buffer_cast<const void*>(buffers);
  }

  static Buffer first(const asio::const_buffers_1& buffers)
  {
    return Buffer(buffers);
  }

private:
  native_buffer_type buffer_;
  std::size_t total_buffer_size_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_BUFFER_SEQUENCE_ADAPTER_HPP
