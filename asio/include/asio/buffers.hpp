//
// buffers.hpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BUFFERS_HPP
#define ASIO_BUFFERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffer.hpp"

namespace asio {

/// Holds a list of buffers that cannot be modified.
/**
 * The const_buffers class template provides a list of buffers that can be
 * passed to operations expecting an implementation of the Const_Buffers
 * concept. It does not own the underlying data, and so is cheap to copy or
 * assign.
 *
 * Instances of this class template may be created using the asio::buffers
 * function:
 *
 * @code sock.write(buffers(buffer1, length1)(buffer2, length2)); @endcode
 *
 * or:
 *
 * @code asio::const_buffers<2> my_buffers =
 *   buffers(buffer1, length1)(buffer2, length2);
 * sock.send(my_buffers); @endcode
 *
 * It may also be explicitly instantiated and initialised using array
 * initialisation syntax:
 *
 * @code asio::const_buffers<2> my_buffers =
 * {
 *   asio::buffer(buffer1, length1),
 *   asio::buffer(buffer2, length2)
 * }; @endcode
 *
 * The const_buffers class template supports conversion to standard containers,
 * where the list of buffers may be manipulated:
 *
 * @code std::vector<asio::const_buffer> my_buffers =
 *   buffers(buffer1, length1)(buffer2, length2);
 * // ...
 * sock.send(my_buffers); @endcode
 */
template <std::size_t N>
class const_buffers
{
public:
  /// The type for each element in the list of buffers.
  typedef const_buffer value_type;

  /// A random-access iterator type that may be used to read or modify elements.
  typedef const_buffer* iterator;

  /// A random-access iterator type that may be used to read elements.
  typedef const const_buffer* const_iterator;

  /// The type for a reference to an element in the list of buffers.
  typedef const_buffer& reference;

  /// The type for a constant reference to an element in the list of buffers.
  typedef const const_buffer& const_reference;

  /// The type used to count the number of buffers in the list.
  typedef std::size_t size_type;

  /// The type used for the difference in addresses of two list elements.
  typedef std::ptrdiff_t difference_type;

  /// Get a random-access iterator to the first element.
  iterator begin()
  {
    return buffers;
  }

  /// Get a random-access iterator to the first element.
  const_iterator begin() const
  {
    return buffers;
  }

  /// Get a random-access iterator for one past the last element.
  iterator end()
  {
    return buffers + N;
  }

  /// Get a random-access iterator for one past the last element.
  const_iterator end() const
  {
    return buffers + N;
  }

  /// Get the number of buffers.
  size_type size() const
  {
    return N;
  }

  /// Get the buffer at the specified index.
  reference operator[](size_type i)
  {
    return buffers[i];
  }

  /// Get the buffer at the specified index.
  const_reference operator[](size_type i) const
  {
    return buffers[i];
  }

  /// Create a new const_buffers instance with one additional element.
  const_buffers<N + 1> operator()(const const_buffer& b)
  {
    const_buffers<N + 1> tmp;
    for (std::size_t i = 0; i < N; ++i)
      tmp.buffers[i] = buffers[i];
    tmp.buffers[N] = b;
    return tmp;
  }

  /// Create a new const_buffers instance with one additional element.
  const_buffers<N + 1> operator()(const void* data, std::size_t size)
  {
    return operator()(buffer(data, size));
  }

  /// Convert to a container.
  template <typename Container>
  operator Container() const
  {
    return Container(begin(), end());
  }

  const_buffer buffers[N];
};

/// Holds a list of buffers that can be modified.
/**
 * The mutable_buffers class template provides a list of buffers that can be
 * passed to operations expecting an implementation of the Mutable_Buffers
 * concept. It does not own the underlying data, and so is cheap to copy or
 * assign.
 *
 * Instances of this class template may be created using the asio::buffers
 * function:
 *
 * @code sock.read(buffers(buffer1, length1)(buffer2, length2)); @endcode
 *
 * or:
 *
 * @code asio::mutable_buffers<2> my_buffers =
 *   buffers(buffer1, length1)(buffer2, length2);
 * sock.send(my_buffers); @endcode
 *
 * It may also be explicitly instantiated and initialised using array
 * initialisation syntax:
 *
 * @code asio::mutable_buffers<2> my_buffers =
 * {
 *   asio::buffer(buffer1, length1),
 *   asio::buffer(buffer2, length2)
 * }; @endcode
 *
 * The mutable_buffers class template supports conversion to standard
 * containers, where the list of buffers may be manipulated:
 *
 * @code std::vector<asio::mutable_buffer> my_buffers =
 *   buffers(buffer1, length1)(buffer2, length2);
 * // ...
 * sock.send(my_buffers); @endcode
 *
 * A mutable_buffers instance may be converted into a corresponding
 * const_buffers instance:
 *
 * @code asio::mutable_buffers<2> my_buffers = ...;
 * asio::const_buffers<2> my_const_buffers = my_buffers; @endcode
 */
template <std::size_t N>
class mutable_buffers
{
public:
  /// The type for each element in the list of buffers.
  typedef mutable_buffer value_type;

  /// A random-access iterator type that may be used to read or modify elements.
  typedef mutable_buffer* iterator;

  /// A random-access iterator type that may be used to read elements.
  typedef const mutable_buffer* const_iterator;

  /// The type for a reference to an element in the list of buffers.
  typedef mutable_buffer& reference;

  /// The type for a constant reference to an element in the list of buffers.
  typedef const mutable_buffer& const_reference;

  /// The type used to count the number of buffers in the list.
  typedef std::size_t size_type;

  /// The type used for the difference in addresses of two list elements.
  typedef ptrdiff_t difference_type;

  /// Get a random-access iterator to the first element.
  iterator begin()
  {
    return buffers;
  }

  /// Get a random-access iterator to the first element.
  const_iterator begin() const
  {
    return buffers;
  }

  /// Get a random-access iterator for one past the last element.
  iterator end()
  {
    return buffers + N;
  }

  /// Get a random-access iterator for one past the last element.
  const_iterator end() const
  {
    return buffers + N;
  }

  /// Get the number of buffers.
  size_type size() const
  {
    return N;
  }

  /// Get the buffer at the specified index.
  reference operator[](size_type i)
  {
    return buffers[i];
  }

  /// Get the buffer at the specified index.
  const_reference operator[](size_type i) const
  {
    return buffers[i];
  }

  /// Create a new mutable_buffers instance with one additional element.
  mutable_buffers<N + 1> operator()(const mutable_buffer& b)
  {
    mutable_buffers<N + 1> tmp;
    for (std::size_t i = 0; i < N; ++i)
      tmp.buffers[i] = buffers[i];
    tmp.buffers[N] = b;
    return tmp;
  }

  /// Create a new const_buffers instance with one additional element.
  const_buffers<N + 1> operator()(const const_buffer& b)
  {
    const_buffers<N + 1> tmp;
    for (std::size_t i = 0; i < N; ++i)
      tmp.buffers[i] = buffers[i];
    tmp.buffers[N] = b;
    return tmp;
  }

  /// Create a new mutable_buffers instance with one additional element.
  mutable_buffers<N + 1> operator()(void* data, std::size_t size)
  {
    return operator()(buffer(data, size));
  }

  /// Create a new const_buffers instance with one additional element.
  const_buffers<N + 1> operator()(const void* data, std::size_t size)
  {
    return operator()(buffer(data, size));
  }

  /// Convert to a const_buffers instance.
  operator const_buffers<N>() const
  {
    const_buffers<N> tmp;
    for (std::size_t i = 0; i < N; ++i)
      tmp.buffers[i] = buffers[i];
    return tmp;
  }

  /// Convert to a container.
  template <typename Container>
  operator Container() const
  {
    return Container(begin(), end());
  }

  mutable_buffer buffers[N];
};

/// Create a const_buffers instance with one element.
inline const_buffers<1> buffers(const const_buffer& b)
{
  const_buffers<1> tmp = { b };
  return tmp;
}

/// Create a mutable_buffers instance with one element.
inline mutable_buffers<1> buffers(const mutable_buffer& b)
{
  mutable_buffers<1> tmp = { b };
  return tmp;
}

/// Create a const_buffers instance with one element.
inline const_buffers<1> buffers(const void* data, std::size_t size)
{
  return buffers(buffer(data, size));
}

/// Create a mutable_buffers instance with one element.
inline mutable_buffers<1> buffers(void* data, std::size_t size)
{
  return buffers(buffer(data, size));
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERS_HPP
