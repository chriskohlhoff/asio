//
// fixed_buffer.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_FIXED_BUFFER_HPP
#define ASIO_FIXED_BUFFER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cassert>
#include <cstring>
#include "asio/detail/pop_options.hpp"

namespace asio {

/// The fixed_buffer class template can be used as a byte buffer.
template <int Buffer_Size>
class fixed_buffer
{
public:
  /// The type of the bytes stored in the buffer.
  typedef char byte_type;

  /// Iterator type for this container.
  typedef byte_type* iterator;

  /// Constant iterator type for this container.
  typedef const byte_type* const_iterator;

  /// The type used for offsets into the buffer.
  typedef size_t size_type;

  /// Constructor.
  fixed_buffer()
    : begin_offset_(0),
      end_offset_(0)
  {
  }

  /// Clear the buffer.
  void clear()
  {
    begin_offset_ = 0;
    end_offset_ = 0;
  }

  /// Return a pointer to the beginning of the unread data.
  iterator begin()
  {
    return buffer_ + begin_offset_;
  }

  /// Return a pointer to the beginning of the unread data.
  const_iterator begin() const
  {
    return buffer_ + begin_offset_;
  }

  /// Get the byte at the front of the buffer.
  byte_type& front()
  {
    return buffer_[begin_offset_];
  }

  /// Get the byte at the front of the buffer.
  const byte_type& front() const
  {
    return buffer_[begin_offset_];
  }

  /// Return a pointer to one past the end of the unread data.
  iterator end()
  {
    return buffer_ + end_offset_;
  }

  /// Return a pointer to one past the end of the unread data.
  const_iterator end() const
  {
    return buffer_ + end_offset_;
  }

  /// Get the byte at the back of the buffer.
  byte_type& back()
  {
    assert(!empty());
    return buffer_[end_offset_ - 1];
  }

  /// Get the byte at the back of the buffer.
  const byte_type& back() const
  {
    assert(!empty());
    return buffer_[end_offset_ - 1];
  }

  /// Get the byte at the given offset in the buffer.
  byte_type& operator[](size_type offset)
  {
    assert(offset < size());
    return buffer_[begin_offset_ + offset];
  }

  /// Get the byte at the given offset in the buffer.
  const byte_type& operator[](size_type offset) const
  {
    assert(offset < size());
    return buffer_[begin_offset_ + offset];
  }

  /// Is there no unread data in the buffer.
  bool empty() const
  {
    return begin_offset_ == end_offset_;
  }

  /// Return the amount of unread data the is in the buffer.
  size_type size() const
  {
    return end_offset_ - begin_offset_;
  }

  /// Resize the buffer to the specified length.
  void resize(size_type length)
  {
    assert(length <= Buffer_Size);
    if (begin_offset_ + length <= Buffer_Size)
    {
      end_offset_ = begin_offset_ + length;
    }
    else
    {
      using namespace std; // For memmove.
      memmove(buffer_, buffer_ + begin_offset_, size());
      end_offset_ = length;
      begin_offset_ = 0;
    }
  }

  /// Return the maximum size for data in the buffer.
  size_type capacity() const
  {
    return Buffer_Size;
  }

  /// Pop a single byte from the beginning of the buffer.
  void pop()
  {
    assert(begin_offset_ < end_offset_);
    ++begin_offset_;
    if (empty())
      clear();
  }

  /// Pop multiple bytes from the beginning of the buffer.
  void pop(size_type count)
  {
    assert(begin_offset_ + count <= end_offset_);
    begin_offset_ += count;
    if (empty())
      clear();
  }

  /// Push a single byte on to the end of the buffer.
  void push(const byte_type& b)
  {
    resize(size() + 1);
    back() = b;
  }

  /// Push the same byte on to the buffer a certain number of times.
  void push(const byte_type& b, size_t count)
  {
    resize(size() + count);
    using namespace std; // For memset.
    memset(end() - count, b, count);
  }

private:
  /// The data in the buffer.
  byte_type buffer_[Buffer_Size];

  /// The offset to the beginning of the unread data.
  size_type begin_offset_;

  /// The offset to the end of the unread data.
  size_type end_offset_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_FIXED_BUFFER_HPP
