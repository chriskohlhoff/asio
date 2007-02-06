//
// buffer.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include <boost/type_traits/is_const.hpp>
#include <string>
#include <vector>
#include "asio/detail/pop_options.hpp"

#if defined(BOOST_MSVC)
# if defined(_HAS_ITERATOR_DEBUGGING)
#  if !defined(ASIO_DISABLE_BUFFER_DEBUGGING)
#   define ASIO_ENABLE_BUFFER_DEBUGGING
#  endif // !defined(ASIO_DISABLE_BUFFER_DEBUGGING)
# endif // defined(_HAS_ITERATOR_DEBUGGING)
#endif // defined(BOOST_MSVC)

#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
# include "asio/detail/push_options.hpp"
# include <boost/function.hpp>
# include "asio/detail/pop_options.hpp"
#endif // ASIO_ENABLE_BUFFER_DEBUGGING

namespace asio {

class mutable_buffer;
class const_buffer;

namespace detail {
void* buffer_cast_helper(const mutable_buffer&);
const void* buffer_cast_helper(const const_buffer&);
std::size_t buffer_size_helper(const mutable_buffer&);
std::size_t buffer_size_helper(const const_buffer&);
} // namespace detail

/// Holds a buffer that can be modified.
/**
 * The mutable_buffer class provides a safe representation of a buffer that can
 * be modified. It does not own the underlying data, and so is cheap to copy or
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

#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
  mutable_buffer(void* data, std::size_t size,
      boost::function<void()> debug_check)
    : data_(data),
      size_(size),
      debug_check_(debug_check)
  {
  }

  const boost::function<void()>& get_debug_check() const
  {
    return debug_check_;
  }
#endif // ASIO_ENABLE_BUFFER_DEBUGGING

private:
  friend void* asio::detail::buffer_cast_helper(
      const mutable_buffer& b);
  friend std::size_t asio::detail::buffer_size_helper(
      const mutable_buffer& b);

  void* data_;
  std::size_t size_;

#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
  boost::function<void()> debug_check_;
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
};

namespace detail {

inline void* buffer_cast_helper(const mutable_buffer& b)
{
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
  if (b.debug_check_)
    b.debug_check_();
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
  return b.data_;
}

inline std::size_t buffer_size_helper(const mutable_buffer& b)
{
  return b.size_;
}

} // namespace detail

/// Cast a non-modifiable buffer to a specified pointer to POD type.
/**
 * @relates mutable_buffer
 */
template <typename PointerToPodType>
inline PointerToPodType buffer_cast(const mutable_buffer& b)
{
  return static_cast<PointerToPodType>(detail::buffer_cast_helper(b));
}

/// Get the number of bytes in a non-modifiable buffer.
/**
 * @relates mutable_buffer
 */
inline std::size_t buffer_size(const mutable_buffer& b)
{
  return detail::buffer_size_helper(b);
}

/// Create a new modifiable buffer that is offset from the start of another.
/**
 * @relates mutable_buffer
 */
inline mutable_buffer operator+(const mutable_buffer& b, std::size_t start)
{
  if (start > buffer_size(b))
    return mutable_buffer();
  char* new_data = buffer_cast<char*>(b) + start;
  std::size_t new_size = buffer_size(b) - start;
  return mutable_buffer(new_data, new_size
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
      , b.get_debug_check()
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
      );
}

/// Create a new modifiable buffer that is offset from the start of another.
/**
 * @relates mutable_buffer
 */
inline mutable_buffer operator+(std::size_t start, const mutable_buffer& b)
{
  if (start > buffer_size(b))
    return mutable_buffer();
  char* new_data = buffer_cast<char*>(b) + start;
  std::size_t new_size = buffer_size(b) - start;
  return mutable_buffer(new_data, new_size
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
      , b.get_debug_check()
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
      );
}

/// Adapts a single modifiable buffer so that it meets the requirements of the
/// MutableBufferSequence concept.
class mutable_buffers_1
  : public mutable_buffer
{
public:
  /// The type for each element in the list of buffers.
  typedef mutable_buffer value_type;

  /// A random-access iterator type that may be used to read elements.
  typedef const mutable_buffer* const_iterator;

  /// Construct to represent a single modifiable buffer.
  explicit mutable_buffers_1(const mutable_buffer& b)
    : mutable_buffer(b)
  {
  }

  /// Get a random-access iterator to the first element.
  const_iterator begin() const
  {
    return this;
  }

  /// Get a random-access iterator for one past the last element.
  const_iterator end() const
  {
    return begin() + 1;
  }
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
    : data_(asio::detail::buffer_cast_helper(b)),
      size_(asio::detail::buffer_size_helper(b))
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
      , debug_check_(b.get_debug_check())
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
  {
  }

#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
  const_buffer(const void* data, std::size_t size,
      boost::function<void()> debug_check)
    : data_(data),
      size_(size),
      debug_check_(debug_check)
  {
  }

  const boost::function<void()>& get_debug_check() const
  {
    return debug_check_;
  }
#endif // ASIO_ENABLE_BUFFER_DEBUGGING

private:
  friend const void* asio::detail::buffer_cast_helper(
      const const_buffer& b);
  friend std::size_t asio::detail::buffer_size_helper(
      const const_buffer& b);

  const void* data_;
  std::size_t size_;

#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
  boost::function<void()> debug_check_;
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
};

namespace detail {

inline const void* buffer_cast_helper(const const_buffer& b)
{
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
  if (b.debug_check_)
    b.debug_check_();
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
  return b.data_;
}

inline std::size_t buffer_size_helper(const const_buffer& b)
{
  return b.size_;
}

} // namespace detail

/// Cast a non-modifiable buffer to a specified pointer to POD type.
/**
 * @relates const_buffer
 */
template <typename PointerToPodType>
inline PointerToPodType buffer_cast(const const_buffer& b)
{
  return static_cast<PointerToPodType>(detail::buffer_cast_helper(b));
}

/// Get the number of bytes in a non-modifiable buffer.
/**
 * @relates const_buffer
 */
inline std::size_t buffer_size(const const_buffer& b)
{
  return detail::buffer_size_helper(b);
}

/// Create a new non-modifiable buffer that is offset from the start of another.
/**
 * @relates const_buffer
 */
inline const_buffer operator+(const const_buffer& b, std::size_t start)
{
  if (start > buffer_size(b))
    return const_buffer();
  const char* new_data = buffer_cast<const char*>(b) + start;
  std::size_t new_size = buffer_size(b) - start;
  return const_buffer(new_data, new_size
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
      , b.get_debug_check()
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
      );
}

/// Create a new non-modifiable buffer that is offset from the start of another.
/**
 * @relates const_buffer
 */
inline const_buffer operator+(std::size_t start, const const_buffer& b)
{
  if (start > buffer_size(b))
    return const_buffer();
  const char* new_data = buffer_cast<const char*>(b) + start;
  std::size_t new_size = buffer_size(b) - start;
  return const_buffer(new_data, new_size
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
      , b.get_debug_check()
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
      );
}

/// Adapts a single non-modifiable buffer so that it meets the requirements of
/// the ConstBufferSequence concept.
class const_buffers_1
  : public const_buffer
{
public:
  /// The type for each element in the list of buffers.
  typedef const_buffer value_type;

  /// A random-access iterator type that may be used to read elements.
  typedef const const_buffer* const_iterator;

  /// Construct to represent a single non-modifiable buffer.
  explicit const_buffers_1(const const_buffer& b)
    : const_buffer(b)
  {
  }

  /// Get a random-access iterator to the first element.
  const_iterator begin() const
  {
    return this;
  }

  /// Get a random-access iterator for one past the last element.
  const_iterator end() const
  {
    return begin() + 1;
  }
};

#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
namespace detail {

template <typename Iterator>
class buffer_debug_check
{
public:
  buffer_debug_check(Iterator iter)
    : iter_(iter)
  {
  }

  void operator()()
  {
    *iter_;
  }

private:
  Iterator iter_;
};

} // namespace detail
#endif // ASIO_ENABLE_BUFFER_DEBUGGING

/** @defgroup buffer asio::buffer
 *
 * @brief The asio::buffer function is used to create a buffer object to
 * represent raw memory, an array of POD elements, or a vector of POD elements.
 *
 * The simplest use case involves reading or writing a single buffer of a
 * specified size:
 *
 * @code sock.write(asio::buffer(data, size)); @endcode
 *
 * In the above example, the return value of asio::buffer meets the
 * requirements of the ConstBufferSequence concept so that it may be directly
 * passed to the socket's write function. A buffer created for modifiable
 * memory also meets the requirements of the MutableBufferSequence concept.
 *
 * An individual buffer may be created from a builtin array, std::vector or
 * boost::array of POD elements. This helps prevent buffer overruns by
 * automatically determining the size of the buffer:
 *
 * @code char d1[128];
 * size_t bytes_transferred = sock.read(asio::buffer(d1));
 *
 * std::vector<char> d2(128);
 * bytes_transferred = sock.read(asio::buffer(d2));
 *
 * boost::array<char, 128> d3;
 * bytes_transferred = sock.read(asio::buffer(d3)); @endcode
 *
 * To read or write using multiple buffers (i.e. scatter-gather I/O), multiple
 * buffer objects may be assigned into a container that supports the
 * MutableBufferSequence (for read) or ConstBufferSequence (for write) concepts:
 *
 * @code
 * char d1[128];
 * std::vector<char> d2(128);
 * boost::array<char, 128> d3;
 *
 * boost::array<mutable_buffer, 3> bufs1 = {
 *   asio::buffer(d1),
 *   asio::buffer(d2),
 *   asio::buffer(d3) };
 * bytes_transferred = sock.read(bufs1);
 *
 * std::vector<const_buffer> bufs2;
 * bufs2.push_back(asio::buffer(d1));
 * bufs2.push_back(asio::buffer(d2));
 * bufs2.push_back(asio::buffer(d3));
 * bytes_transferred = sock.write(bufs2); @endcode
 */
/*@{*/

/// Create a new modifiable buffer from an existing buffer.
inline mutable_buffers_1 buffer(const mutable_buffer& b)
{
  return mutable_buffers_1(b);
}

/// Create a new modifiable buffer from an existing buffer.
inline mutable_buffers_1 buffer(const mutable_buffer& b,
    std::size_t max_size_in_bytes)
{
  return mutable_buffers_1(
      mutable_buffer(buffer_cast<void*>(b),
        buffer_size(b) < max_size_in_bytes
        ? buffer_size(b) : max_size_in_bytes
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        , b.get_debug_check()
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
        ));
}

/// Create a new non-modifiable buffer from an existing buffer.
inline const_buffers_1 buffer(const const_buffer& b)
{
  return const_buffers_1(b);
}

/// Create a new non-modifiable buffer from an existing buffer.
inline const_buffers_1 buffer(const const_buffer& b,
    std::size_t max_size_in_bytes)
{
  return const_buffers_1(
      const_buffer(buffer_cast<const void*>(b),
        buffer_size(b) < max_size_in_bytes
        ? buffer_size(b) : max_size_in_bytes
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        , b.get_debug_check()
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
        ));
}

/// Create a new modifiable buffer that represents the given memory range.
inline mutable_buffers_1 buffer(void* data, std::size_t size_in_bytes)
{
  return mutable_buffers_1(mutable_buffer(data, size_in_bytes));
}

/// Create a new non-modifiable buffer that represents the given memory range.
inline const_buffers_1 buffer(const void* data,
    std::size_t size_in_bytes)
{
  return const_buffers_1(const_buffer(data, size_in_bytes));
}

/// Create a new modifiable buffer that represents the given POD array.
template <typename PodType, std::size_t N>
inline mutable_buffers_1 buffer(PodType (&data)[N])
{
  return mutable_buffers_1(mutable_buffer(data, N * sizeof(PodType)));
}
 
/// Create a new modifiable buffer that represents the given POD array.
template <typename PodType, std::size_t N>
inline mutable_buffers_1 buffer(PodType (&data)[N],
    std::size_t max_size_in_bytes)
{
  return mutable_buffers_1(
      mutable_buffer(data,
        N * sizeof(PodType) < max_size_in_bytes
        ? N * sizeof(PodType) : max_size_in_bytes));
}
 
/// Create a new non-modifiable buffer that represents the given POD array.
template <typename PodType, std::size_t N>
inline const_buffers_1 buffer(const PodType (&data)[N])
{
  return const_buffers_1(const_buffer(data, N * sizeof(PodType)));
}

/// Create a new non-modifiable buffer that represents the given POD array.
template <typename PodType, std::size_t N>
inline const_buffers_1 buffer(const PodType (&data)[N],
    std::size_t max_size_in_bytes)
{
  return const_buffers_1(
      const_buffer(data,
        N * sizeof(PodType) < max_size_in_bytes
        ? N * sizeof(PodType) : max_size_in_bytes));
}

#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x582))

// Borland C++ thinks the overloads:
//
//   unspecified buffer(boost::array<PodType, N>& array ...);
//
// and
//
//   unspecified buffer(boost::array<const PodType, N>& array ...);
//
// are ambiguous. This will be worked around by using a buffer_types traits
// class that contains typedefs for the appropriate buffer and container
// classes, based on whether PodType is const or non-const.

namespace detail {

template <bool IsConst>
struct buffer_types_base;

template <>
struct buffer_types_base<false>
{
  typedef mutable_buffer buffer_type;
  typedef mutable_buffers_1 container_type;
};

template <>
struct buffer_types_base<true>
{
  typedef const_buffer buffer_type;
  typedef const_buffers_1 container_type;
};

template <typename PodType>
struct buffer_types
  : public buffer_types_base<boost::is_const<PodType>::value>
{
};

} // namespace detail

template <typename PodType, std::size_t N>
inline typename detail::buffer_types<PodType>::container_type
buffer(boost::array<PodType, N>& data)
{
  typedef typename asio::detail::buffer_types<PodType>::buffer_type
    buffer_type;
  typedef typename asio::detail::buffer_types<PodType>::container_type
    container_type;
  return container_type(
      buffer_type(data.c_array(), data.size() * sizeof(PodType)));
}

template <typename PodType, std::size_t N>
inline typename detail::buffer_types<PodType>::container_type
buffer(boost::array<PodType, N>& data, std::size_t max_size_in_bytes)
{
  typedef typename asio::detail::buffer_types<PodType>::buffer_type
    buffer_type;
  typedef typename asio::detail::buffer_types<PodType>::container_type
    container_type;
  return container_type(
      buffer_type(data.c_array(),
        data.size() * sizeof(PodType) < max_size_in_bytes
        ? data.size() * sizeof(PodType) : max_size_in_bytes));
}

#else // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x582))

/// Create a new modifiable buffer that represents the given POD array.
template <typename PodType, std::size_t N>
inline mutable_buffers_1 buffer(boost::array<PodType, N>& data)
{
  return mutable_buffers_1(
      mutable_buffer(data.c_array(), data.size() * sizeof(PodType)));
}

/// Create a new modifiable buffer that represents the given POD array.
template <typename PodType, std::size_t N>
inline mutable_buffers_1 buffer(boost::array<PodType, N>& data,
    std::size_t max_size_in_bytes)
{
  return mutable_buffers_1(
      mutable_buffer(data.c_array(),
        data.size() * sizeof(PodType) < max_size_in_bytes
        ? data.size() * sizeof(PodType) : max_size_in_bytes));
}

/// Create a new non-modifiable buffer that represents the given POD array.
template <typename PodType, std::size_t N>
inline const_buffers_1 buffer(boost::array<const PodType, N>& data)
{
  return const_buffers_1(
      const_buffer(data.data(), data.size() * sizeof(PodType)));
}

/// Create a new non-modifiable buffer that represents the given POD array.
template <typename PodType, std::size_t N>
inline const_buffers_1 buffer(boost::array<const PodType, N>& data,
    std::size_t max_size_in_bytes)
{
  return const_buffers_1(
      const_buffer(data.data(),
        data.size() * sizeof(PodType) < max_size_in_bytes
        ? data.size() * sizeof(PodType) : max_size_in_bytes));
}

#endif // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x582))

/// Create a new non-modifiable buffer that represents the given POD array.
template <typename PodType, std::size_t N>
inline const_buffers_1 buffer(const boost::array<PodType, N>& data)
{
  return const_buffers_1(
      const_buffer(data.data(), data.size() * sizeof(PodType)));
}

/// Create a new non-modifiable buffer that represents the given POD array.
template <typename PodType, std::size_t N>
inline const_buffers_1 buffer(const boost::array<PodType, N>& data,
    std::size_t max_size_in_bytes)
{
  return const_buffers_1(
      const_buffer(data.data(),
        data.size() * sizeof(PodType) < max_size_in_bytes
        ? data.size() * sizeof(PodType) : max_size_in_bytes));
}

/// Create a new modifiable buffer that represents the given POD vector.
/**
 * @note The buffer is invalidated by any vector operation that would also
 * invalidate iterators.
 */
template <typename PodType, typename Allocator>
inline mutable_buffers_1 buffer(std::vector<PodType, Allocator>& data)
{
  return mutable_buffers_1(
      mutable_buffer(&data[0], data.size() * sizeof(PodType)
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        , detail::buffer_debug_check<
            typename std::vector<PodType, Allocator>::iterator
          >(data.begin())
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
        ));
}

/// Create a new modifiable buffer that represents the given POD vector.
/**
 * @note The buffer is invalidated by any vector operation that would also
 * invalidate iterators.
 */
template <typename PodType, typename Allocator>
inline mutable_buffers_1 buffer(std::vector<PodType, Allocator>& data,
    std::size_t max_size_in_bytes)
{
  return mutable_buffers_1(
      mutable_buffer(&data[0],
        data.size() * sizeof(PodType) < max_size_in_bytes
        ? data.size() * sizeof(PodType) : max_size_in_bytes
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        , detail::buffer_debug_check<
            typename std::vector<PodType, Allocator>::iterator
          >(data.begin())
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
        ));
}

/// Create a new non-modifiable buffer that represents the given POD vector.
/**
 * @note The buffer is invalidated by any vector operation that would also
 * invalidate iterators.
 */
template <typename PodType, typename Allocator>
inline const_buffers_1 buffer(
    const std::vector<PodType, Allocator>& data)
{
  return const_buffers_1(
      const_buffer(&data[0], data.size() * sizeof(PodType)
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        , detail::buffer_debug_check<
            typename std::vector<PodType, Allocator>::const_iterator
          >(data.begin())
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
        ));
}

/// Create a new non-modifiable buffer that represents the given POD vector.
/**
 * @note The buffer is invalidated by any vector operation that would also
 * invalidate iterators.
 */
template <typename PodType, typename Allocator>
inline const_buffers_1 buffer(
    const std::vector<PodType, Allocator>& data, std::size_t max_size_in_bytes)
{
  return const_buffers_1(
      const_buffer(&data[0],
        data.size() * sizeof(PodType) < max_size_in_bytes
        ? data.size() * sizeof(PodType) : max_size_in_bytes
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        , detail::buffer_debug_check<
            typename std::vector<PodType, Allocator>::const_iterator
          >(data.begin())
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
        ));
}

/// Create a new non-modifiable buffer that represents the given string.
/**
 * @note The buffer is invalidated by any non-const operation called on the
 * given string object.
 */
inline const_buffers_1 buffer(const std::string& data)
{
  return const_buffers_1(const_buffer(data.data(), data.size()
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        , detail::buffer_debug_check<std::string::const_iterator>(data.begin())
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
        ));
}

/// Create a new non-modifiable buffer that represents the given string.
/**
 * @note The buffer is invalidated by any non-const operation called on the
 * given string object.
 */
inline const_buffers_1 buffer(const std::string& data,
    std::size_t max_size_in_bytes)
{
  return const_buffers_1(
      const_buffer(data.data(),
        data.size() < max_size_in_bytes
        ? data.size() : max_size_in_bytes
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        , detail::buffer_debug_check<std::string::const_iterator>(data.begin())
#endif // ASIO_ENABLE_BUFFER_DEBUGGING
        ));
}

/*@}*/

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFER_HPP
