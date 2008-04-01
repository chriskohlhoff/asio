//
// consuming_buffers.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_CONSUMING_BUFFERS_HPP
#define ASIO_DETAIL_CONSUMING_BUFFERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <algorithm>
#include <cstddef>
#include <boost/config.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffer.hpp"

namespace asio {
namespace detail {

// A proxy iterator for a sub-range in a list of buffers.
template <typename Buffer, typename Buffer_Iterator>
class consuming_buffers_iterator
  : public boost::iterator_facade<
        consuming_buffers_iterator<Buffer, Buffer_Iterator>,
        const Buffer, boost::forward_traversal_tag>
{
public:
  // Default constructor creates an end iterator.
  consuming_buffers_iterator()
    : at_end_(true)
  {
  }

  // Construct with a buffer for the first entry and an iterator
  // range for the remaining entries.
  consuming_buffers_iterator(bool at_end, const Buffer& first,
      Buffer_Iterator begin_remainder, Buffer_Iterator end_remainder)
    : at_end_(at_end),
      first_(buffer(first, max_size)),
      begin_remainder_(begin_remainder),
      end_remainder_(end_remainder),
      offset_(0)
  {
  }

private:
  friend class boost::iterator_core_access;

  enum { max_size = 65536 };

  void increment()
  {
    if (!at_end_)
    {
      if (begin_remainder_ == end_remainder_
          || offset_ + buffer_size(first_) >= max_size)
      {
        at_end_ = true;
      }
      else
      {
        offset_ += buffer_size(first_);
        first_ = buffer(*begin_remainder_++, max_size - offset_);
      }
    }
  }

  bool equal(const consuming_buffers_iterator& other) const
  {
    if (at_end_ && other.at_end_)
      return true;
    return !at_end_ && !other.at_end_
      && buffer_cast<const void*>(first_)
        == buffer_cast<const void*>(other.first_)
      && buffer_size(first_) == buffer_size(other.first_)
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
};

// A proxy for a sub-range in a list of buffers.
template <typename Buffer, typename Buffers>
class consuming_buffers
{
public:
  // The type for each element in the list of buffers.
  typedef Buffer value_type;

  // A forward-only iterator type that may be used to read elements.
  typedef consuming_buffers_iterator<Buffer, typename Buffers::const_iterator>
    const_iterator;

  // Construct to represent the entire list of buffers.
  consuming_buffers(const Buffers& buffers)
    : buffers_(buffers),
      at_end_(buffers_.begin() == buffers_.end()),
      first_(*buffers_.begin()),
      begin_remainder_(buffers_.begin())
  {
    if (!at_end_)
      ++begin_remainder_;
  }

  // Copy constructor.
  consuming_buffers(const consuming_buffers& other)
    : buffers_(other.buffers_),
      at_end_(other.at_end_),
      first_(other.first_),
      begin_remainder_(buffers_.begin())
  {
    typename Buffers::const_iterator first = other.buffers_.begin();
    typename Buffers::const_iterator second = other.begin_remainder_;
    std::advance(begin_remainder_, std::distance(first, second));
  }

  // Assignment operator.
  consuming_buffers& operator=(const consuming_buffers& other)
  {
    buffers_ = other.buffers_;
    at_end_ = other.at_end_;
    first_ = other.first_;
    begin_remainder_ = buffers_.begin();
    typename Buffers::const_iterator first = other.buffers_.begin();
    typename Buffers::const_iterator second = other.begin_remainder_;
    std::advance(begin_remainder_, std::distance(first, second));
    return *this;
  }

  // Get a forward-only iterator to the first element.
  const_iterator begin() const
  {
    return const_iterator(at_end_, first_, begin_remainder_, buffers_.end());
  }

  // Get a forward-only iterator for one past the last element.
  const_iterator end() const
  {
    return const_iterator();
  }

  // Consume the specified number of bytes from the buffers.
  void consume(std::size_t size)
  {
    // Remove buffers from the start until the specified size is reached.
    while (size > 0 && !at_end_)
    {
      if (buffer_size(first_) <= size)
      {
        size -= buffer_size(first_);
        if (begin_remainder_ == buffers_.end())
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
    while (!at_end_ && buffer_size(first_) == 0)
    {
      if (begin_remainder_ == buffers_.end())
        at_end_ = true;
      else
        first_ = *begin_remainder_++;
    }
  }

private:
  Buffers buffers_;
  bool at_end_;
  Buffer first_;
  typename Buffers::const_iterator begin_remainder_;
};

// Specialisation for null_buffers to ensure that the null_buffers type is
// always passed through to the underlying read or write operation.
template <typename Buffer>
class consuming_buffers<Buffer, asio::null_buffers>
  : public asio::null_buffers
{
public:
  consuming_buffers(const asio::null_buffers&)
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
