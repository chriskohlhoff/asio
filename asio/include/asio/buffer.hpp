//
// buffer.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BUFFER_HPP
#define ASIO_BUFFER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {

/// Holds a buffer that can be modified.
/**
 * The const_buffer class provides a safe representation of a buffer that can be
 * modified. It does not own the underlying data, and so is cheap to copy or
 * assign.
 */
class mutable_buffer
{
public:
  /// Construct an empty buffer.
  mutable_buffer()
    : data_(0),
      size_(0)
  {
  }

  /// Construct a buffer to represent a given memory range.
  mutable_buffer(void* data, std::size_t size)
    : data_(data),
      size_(size)
  {
  }

  /// Get a pointer to the start of the buffer's memory.
  void* data() const
  {
    return data_;
  }

  /// Get the size of the buffer, in bytes.
  std::size_t size() const
  {
    return size_;
  }

  /// Obtain a new buffer that represents a part of the buffer.
  mutable_buffer sub_buffer(std::size_t start) const
  {
    if (start > size_)
      return mutable_buffer();
    char* new_data = static_cast<char*>(data_) + start;
    std::size_t new_size = size_ - start;
    return mutable_buffer(new_data, new_size);
  }

  /// Obtain a new buffer that represents a part of the buffer.
  mutable_buffer sub_buffer(std::size_t start, std::size_t size) const
  {
    if (start > size_)
      return mutable_buffer();
    char* new_data = static_cast<char*>(data_) + start;
    std::size_t new_size = (size_ - start < size) ? (size_ - start) : size;
    return mutable_buffer(new_data, new_size);
  }

private:
  void* data_;
  std::size_t size_;
};

/// Holds a buffer that cannot be modified.
/**
 * The const_buffer class provides a safe representation of a buffer that cannot
 * be modified. It does not own the underlying data, and so is cheap to copy or
 * assign.
 */
class const_buffer
{
public:
  /// Construct an empty buffer.
  const_buffer()
    : data_(0),
      size_(0)
  {
  }

  /// Construct a buffer to represent a given memory range.
  const_buffer(const void* data, std::size_t size)
    : data_(data),
      size_(size)
  {
  }

  /// Construct a non-modifiable buffer from a modifiable one.
  const_buffer(const mutable_buffer& b)
    : data_(b.data()),
      size_(b.size())
  {
  }

  /// Get a pointer to the start of the buffer's memory.
  const void* data() const
  {
    return data_;
  }

  /// Get the size of the buffer, in bytes.
  std::size_t size() const
  {
    return size_;
  }

  /// Obtain a new buffer that represents a part of the buffer.
  const_buffer sub_buffer(std::size_t start) const
  {
    if (start > size_)
      return const_buffer();
    const char* new_data = static_cast<const char*>(data_) + start;
    std::size_t new_size = size_ - start;
    return const_buffer(new_data, new_size);
  }

  /// Obtain a new buffer that represents a part of the buffer.
  const_buffer sub_buffer(std::size_t start, std::size_t size) const
  {
    if (start > size_)
      return const_buffer();
    const char* new_data = static_cast<const char*>(data_) + start;
    std::size_t new_size = (size_ - start < size) ? (size_ - start) : size;
    return const_buffer(new_data, new_size);
  }

private:
  const void* data_;
  std::size_t size_;
};

/// Create a new modifiable buffer that represents the given memory range.
inline mutable_buffer buffer(void* data, std::size_t size)
{
  return mutable_buffer(data, size);
}

/// Create a new non-modifiable buffer that represents the given memory range.
inline const_buffer buffer(const void* data, std::size_t size)
{
  return const_buffer(data, size);
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFER_HPP
