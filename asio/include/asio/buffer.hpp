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
#include <boost/array.hpp>
#include <vector>
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

private:
  void* data_;
  std::size_t size_;
};

/// Create a new modifiable buffer that is offset from the start of another.
inline mutable_buffer operator+(const mutable_buffer& b, std::size_t start)
{
  if (start > b.size())
    return mutable_buffer();
  char* new_data = static_cast<char*>(b.data()) + start;
  std::size_t new_size = b.size() - start;
  return mutable_buffer(new_data, new_size);
}

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

private:
  const void* data_;
  std::size_t size_;
};

/// Create a new non-modifiable buffer that is offset from the start of another.
inline const_buffer operator+(const const_buffer& b, std::size_t start)
{
  if (start > b.size())
    return const_buffer();
  const char* new_data = static_cast<const char*>(b.data()) + start;
  std::size_t new_size = b.size() - start;
  return const_buffer(new_data, new_size);
}

/**
 * @defgroup buffer asio::buffer
 */
/*@{*/

/// Create a new modifiable buffer from an existing buffer.
inline mutable_buffer buffer(const mutable_buffer& b)
{
  return mutable_buffer(b);
}

/// Create a new modifiable buffer from an existing buffer.
inline mutable_buffer buffer(const mutable_buffer& b,
    std::size_t max_size_in_bytes)
{
  return mutable_buffer(b.data(),
      b.size() < max_size_in_bytes ? b.size() : max_size_in_bytes);
}

/// Create a new non-modifiable buffer from an existing buffer.
inline const_buffer buffer(const const_buffer& b)
{
  return const_buffer(b);
}

/// Create a new non-modifiable buffer from an existing buffer.
inline const_buffer buffer(const const_buffer& b,
    std::size_t max_size_in_bytes)
{
  return const_buffer(b.data(),
      b.size() < max_size_in_bytes ? b.size() : max_size_in_bytes);
}

/// Create a new modifiable buffer that represents the given memory range.
inline mutable_buffer buffer(void* data, std::size_t size_in_bytes)
{
  return mutable_buffer(data, size_in_bytes);
}

/// Create a new non-modifiable buffer that represents the given memory range.
inline const_buffer buffer(const void* data, std::size_t size_in_bytes)
{
  return const_buffer(data, size_in_bytes);
}

/// Create a new modifiable buffer that represents the given POD array.
template <typename Pod_Type, std::size_t N>
inline mutable_buffer buffer(Pod_Type (&data)[N])
{
  return mutable_buffer(data, N * sizeof(Pod_Type));
}
 
/// Create a new modifiable buffer that represents the given POD array.
template <typename Pod_Type, std::size_t N>
inline mutable_buffer buffer(Pod_Type (&data)[N], std::size_t max_size_in_bytes)
{
  return mutable_buffer(data,
      N * sizeof(Pod_Type) < max_size_in_bytes
      ? N * sizeof(Pod_Type) : max_size_in_bytes);
}
 
/// Create a new non-modifiable buffer that represents the given POD array.
template <typename Pod_Type, std::size_t N>
inline const_buffer buffer(const Pod_Type (&data)[N])
{
  return const_buffer(data, N * sizeof(Pod_Type));
}

/// Create a new non-modifiable buffer that represents the given POD array.
template <typename Pod_Type, std::size_t N>
inline const_buffer buffer(const Pod_Type (&data)[N],
    std::size_t max_size_in_bytes)
{
  return const_buffer(data,
      N * sizeof(Pod_Type) < max_size_in_bytes
      ? N * sizeof(Pod_Type) : max_size_in_bytes);
}

/// Create a new modifiable buffer that represents the given POD array.
template <typename Pod_Type, std::size_t N>
inline mutable_buffer buffer(boost::array<Pod_Type, N>& data)
{
  return mutable_buffer(data.c_array(), data.size() * sizeof(Pod_Type));
}

/// Create a new modifiable buffer that represents the given POD array.
template <typename Pod_Type, std::size_t N>
inline mutable_buffer buffer(boost::array<Pod_Type, N>& data,
    std::size_t max_size_in_bytes)
{
  return mutable_buffer(data.c_array(),
      data.size() * sizeof(Pod_Type) < max_size_in_bytes
      ? data.size() * sizeof(Pod_Type) : max_size_in_bytes);
}

/// Create a new non-modifiable buffer that represents the given POD array.
template <typename Pod_Type, std::size_t N>
inline const_buffer buffer(const boost::array<Pod_Type, N>& data)
{
  return const_buffer(data.data(), data.size() * sizeof(Pod_Type));
}

/// Create a new non-modifiable buffer that represents the given POD array.
template <typename Pod_Type, std::size_t N>
inline const_buffer buffer(const boost::array<Pod_Type, N>& data,
    std::size_t max_size_in_bytes)
{
  return const_buffer(data.data(),
      data.size() * sizeof(Pod_Type) < max_size_in_bytes
      ? data.size() * sizeof(Pod_Type) : max_size_in_bytes);
}

/// Create a new non-modifiable buffer that represents the given POD array.
template <typename Pod_Type, std::size_t N>
inline const_buffer buffer(boost::array<const Pod_Type, N>& data)
{
  return const_buffer(data.data(), data.size() * sizeof(Pod_Type));
}

/// Create a new non-modifiable buffer that represents the given POD array.
template <typename Pod_Type, std::size_t N>
inline const_buffer buffer(boost::array<const Pod_Type, N>& data,
    std::size_t max_size_in_bytes)
{
  return const_buffer(data.data(),
      data.size() * sizeof(Pod_Type) < max_size_in_bytes
      ? data.size() * sizeof(Pod_Type) : max_size_in_bytes);
}

/// Create a new modifiable buffer that represents the given POD vector.
template <typename Pod_Type>
inline mutable_buffer buffer(std::vector<Pod_Type>& data)
{
  return mutable_buffer(&data[0], data.size() * sizeof(Pod_Type));
}

/// Create a new modifiable buffer that represents the given POD vector.
template <typename Pod_Type>
inline mutable_buffer buffer(std::vector<Pod_Type>& data,
    std::size_t max_size_in_bytes)
{
  return mutable_buffer(&data[0],
      data.size() * sizeof(Pod_Type) < max_size_in_bytes
      ? data.size() * sizeof(Pod_Type) : max_size_in_bytes);
}

/// Create a new non-modifiable buffer that represents the given POD vector.
template <typename Pod_Type>
inline const_buffer buffer(const std::vector<Pod_Type>& data)
{
  return const_buffer(&data[0], data.size() * sizeof(Pod_Type));
}

/// Create a new non-modifiable buffer that represents the given POD vector.
template <typename Pod_Type>
inline const_buffer buffer(const std::vector<Pod_Type>& data,
    std::size_t max_size_in_bytes)
{
  return const_buffer(&data[0],
      data.size() * sizeof(Pod_Type) < max_size_in_bytes
      ? data.size() * sizeof(Pod_Type) : max_size_in_bytes);
}

/// Create a new non-modifiable buffer that represents the given POD vector.
template <typename Pod_Type>
inline const_buffer buffer(std::vector<const Pod_Type>& data)
{
  return const_buffer(&data[0], data.size() * sizeof(Pod_Type));
}

/// Create a new non-modifiable buffer that represents the given POD vector.
template <typename Pod_Type>
inline const_buffer buffer(std::vector<const Pod_Type>& data,
    std::size_t max_size_in_bytes)
{
  return const_buffer(&data[0],
      data.size() * sizeof(Pod_Type) < max_size_in_bytes
      ? data.size() * sizeof(Pod_Type) : max_size_in_bytes);
}

/*@}*/

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFER_HPP
