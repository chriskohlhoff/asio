//
// consuming_buffers.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_CONSUMING_BUFFERS_HPP
#define ASIO_CONSUMING_BUFFERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <algorithm>
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {

/// A proxy for a sub-range in a list of buffers.
template <typename Buffers>
class consuming_buffers
{
public:
  /// The type for each element in the list of buffers.
  typedef typename Buffers::value_type value_type;

  /// A forward-only iterator type that may be used to read or modify elements.
  typedef typename Buffers::iterator iterator;

  /// A forward-only iterator type that may be used to read elements.
  typedef typename Buffers::const_iterator const_iterator;

  /// Construct to represent the entire list of buffers.
  consuming_buffers(const Buffers& buffers)
    : buffers_(buffers),
      begin_(buffers_.begin())
  {
  }

  /// Copy constructor.
  consuming_buffers(const consuming_buffers& other)
    : buffers_(other.buffers_),
      begin_(buffers_.begin())
  {
    const_iterator first = other.buffers_.begin();
    const_iterator second = other.begin_;
    std::advance(begin_, std::distance(first, second));
  }

  /// Assignment operator.
  consuming_buffers& operator=(const consuming_buffers& other)
  {
    buffers_ = other.buffers_;
    begin_ = buffers_.begin();
    const_iterator first = other.buffers_.begin();
    const_iterator second = other.begin_;
    std::advance(begin_, std::distance(first, second));
    return *this;
  }

  /// Get a forward-only iterator to the first element.
  iterator begin()
  {
    return begin_;
  }

  /// Get a forward-only iterator to the first element.
  const_iterator begin() const
  {
    return begin_;
  }

  /// Get a forward-only iterator for one past the last element.
  iterator end()
  {
    return buffers_.end();
  }

  /// Get a forward-only iterator for one past the last element.
  const_iterator end() const
  {
    return buffers_.end();
  }

  /// Consume the specified number of bytes from the buffers.
  void consume(std::size_t size)
  {
    // Remove buffers from the start until the specified size is reached.
    while (size > 0 && begin_ != buffers_.end())
    {
      if (begin_->size() <= size)
      {
        size -= begin_->size();
        *begin_ = value_type();
        ++begin_;
      }
      else
      {
        *begin_ = *begin_ + size;
        size = 0;
      }
    }

    // Remove any more empty buffers at the start.
    while (begin_ != buffers_.end() && begin_->size() == 0)
      ++begin_;
  }

private:
  Buffers buffers_;
  iterator begin_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_CONSUMING_BUFFERS_HPP
