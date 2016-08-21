//
// detail/consuming_buffers.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_CONSUMING_BUFFERS_HPP
#define ASIO_DETAIL_CONSUMING_BUFFERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <cstddef>
#include <iterator>
#include "asio/buffer.hpp"
#include "asio/detail/limits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// A proxy iterator for a sub-range in a list of buffers.
template <typename Buffer, typename Buffer_Iterator>
class consuming_buffers_iterator
{
public:
  /// The type used for the distance between two iterators.
  typedef std::ptrdiff_t difference_type;

  /// The type of the value pointed to by the iterator.
  typedef Buffer value_type;

  /// The type of the result of applying operator->() to the iterator.
  typedef const Buffer* pointer;

  /// The type of the result of applying operator*() to the iterator.
  typedef const Buffer& reference;

  /// The iterator category.
  typedef std::forward_iterator_tag iterator_category;

  // Default constructor creates an end iterator.
  consuming_buffers_iterator()
    : at_end_(true)
  {
  }

  // Construct with a buffer for the first entry and an iterator
  // range for the remaining entries.
  consuming_buffers_iterator(bool at_end, const Buffer& first,
      Buffer_Iterator begin_remainder, Buffer_Iterator end_remainder,
      std::size_t max_size)
    : at_end_(max_size > 0 ? at_end : true),
      first_(buffer(first, max_size)),
      begin_remainder_(begin_remainder),
      end_remainder_(end_remainder),
      offset_(0),
      max_size_(max_size)
  {
  }

  // Dereference an iterator.
  const Buffer& operator*() const
  {
    return dereference();
  }

  // Dereference an iterator.
  const Buffer* operator->() const
  {
    return &dereference();
  }

  // Increment operator (prefix).
  consuming_buffers_iterator& operator++()
  {
    increment();
    return *this;
  }

  // Increment operator (postfix).
  consuming_buffers_iterator operator++(int)
  {
    consuming_buffers_iterator tmp(*this);
    ++*this;
    return tmp;
  }

  // Test two iterators for equality.
  friend bool operator==(const consuming_buffers_iterator& a,
      const consuming_buffers_iterator& b)
  {
    return a.equal(b);
  }

  // Test two iterators for inequality.
  friend bool operator!=(const consuming_buffers_iterator& a,
      const consuming_buffers_iterator& b)
  {
    return !a.equal(b);
  }

private:
  void increment()
  {
    if (!at_end_)
    {
      if (begin_remainder_ == end_remainder_
          || offset_ + first_.size() >= max_size_)
      {
        at_end_ = true;
      }
      else
      {
        offset_ += first_.size();
        first_ = buffer(*begin_remainder_++, max_size_ - offset_);
      }
    }
  }

  bool equal(const consuming_buffers_iterator& other) const
  {
    if (at_end_ && other.at_end_)
      return true;
    return !at_end_ && !other.at_end_
      && first_.data() == other.first_.data()
      && first_.size() == other.first_.size()
      && begin_remainder_ == other.begin_remainder_
      && end_remainder_ == other.end_remainder_;
  }

  const Buffer& dereference() const
  {
    return first_;
  }

  bool at_end_;
  Buffer first_;
  Buffer_Iterator begin_remainder_;
  Buffer_Iterator end_remainder_;
  std::size_t offset_;
  std::size_t max_size_;
};

// A proxy for a sub-range in a list of buffers.
template <typename Buffer, typename Buffers, typename Buffer_Iterator>
class consuming_buffers
{
public:
  // The type for each element in the list of buffers.
  typedef Buffer value_type;

  // A forward-only iterator type that may be used to read elements.
  typedef consuming_buffers_iterator<Buffer, Buffer_Iterator>
    const_iterator;

  // Construct to represent the entire list of buffers.
  consuming_buffers(const Buffers& buffers)
    : buffers_(buffers),
      at_end_(buffer_sequence_begin(buffers_) == buffer_sequence_end(buffers_)),
      begin_remainder_(buffer_sequence_begin(buffers_)),
      max_size_((std::numeric_limits<std::size_t>::max)())
  {
    if (!at_end_)
    {
      first_ = *buffer_sequence_begin(buffers_);
      ++begin_remainder_;
    }
  }

  // Copy constructor.
  consuming_buffers(const consuming_buffers& other)
    : buffers_(other.buffers_),
      at_end_(other.at_end_),
      first_(other.first_),
      begin_remainder_(buffer_sequence_begin(buffers_)),
      max_size_(other.max_size_)
  {
    Buffer_Iterator first = buffer_sequence_begin(other.buffers_);
    Buffer_Iterator second = other.begin_remainder_;
    std::advance(begin_remainder_, std::distance(first, second));
  }

  // Assignment operator.
  consuming_buffers& operator=(const consuming_buffers& other)
  {
    buffers_ = other.buffers_;
    at_end_ = other.at_end_;
    first_ = other.first_;
    begin_remainder_ = buffer_sequence_begin(buffers_);
    Buffer_Iterator first = buffer_sequence_begin(other.buffers_);
    Buffer_Iterator second = other.begin_remainder_;
    std::advance(begin_remainder_, std::distance(first, second));
    max_size_ = other.max_size_;
    return *this;
  }

  // Get a forward-only iterator to the first element.
  const_iterator begin() const
  {
    return const_iterator(at_end_, first_,
        begin_remainder_, buffer_sequence_end(buffers_), max_size_);
  }

  // Get a forward-only iterator for one past the last element.
  const_iterator end() const
  {
    return const_iterator();
  }

  // Set the maximum size for a single transfer.
  void prepare(std::size_t max_size)
  {
    max_size_ = max_size;
  }

  // Consume the specified number of bytes from the buffers.
  void consume(std::size_t size)
  {
    // Remove buffers from the start until the specified size is reached.
    while (size > 0 && !at_end_)
    {
      if (first_.size() <= size)
      {
        size -= first_.size();
        if (begin_remainder_ == buffer_sequence_end(buffers_))
          at_end_ = true;
        else
          first_ = *begin_remainder_++;
      }
      else
      {
        first_ = first_ + size;
        size = 0;
      }
    }

    // Remove any more empty buffers at the start.
    while (!at_end_ && first_.size() == 0)
    {
      if (begin_remainder_ == buffer_sequence_end(buffers_))
        at_end_ = true;
      else
        first_ = *begin_remainder_++;
    }
  }

private:
  Buffers buffers_;
  bool at_end_;
  Buffer first_;
  Buffer_Iterator begin_remainder_;
  std::size_t max_size_;
};

// Specialisation for null_buffers to ensure that the null_buffers type is
// always passed through to the underlying read or write operation.
template <typename Buffer>
class consuming_buffers<Buffer,
      asio::null_buffers, const mutable_buffer*>
  : public asio::null_buffers
{
public:
  consuming_buffers(const asio::null_buffers&)
  {
    // No-op.
  }

  void prepare(std::size_t)
  {
    // No-op.
  }

  void consume(std::size_t)
  {
    // No-op.
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_CONSUMING_BUFFERS_HPP
