//
// time.hpp
// ~~~~~~~~
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

#ifndef ASIO_TIME_HPP
#define ASIO_TIME_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/time.hpp"

namespace asio {

/// A simple abstraction for representing a time.
/**
 * The asio::time class is used to represent a time with seconds and
 * microseconds components.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
class time
{
public:
  /// Default constructor.
  time()
    : impl_()
  {
  }

  /// Construct with a particular value for the seconds component.
  time(long seconds)
    : impl_(seconds)
  {
  }

  /// Construct with particular values for the seconds and microseconds
  /// components.
  time(long seconds, long microseconds)
    : impl_(seconds, microseconds)
  {
  }

  /// Get the seconds component of the time.
  long sec() const
  {
    return impl_.sec();
  }

  /// Set the seconds component of the time.
  void sec(long seconds)
  {
    impl_.sec(seconds);
  }

  /// Get the microseconds component of the time.
  long usec() const
  {
    return impl_.usec();
  }

  /// Set the microseconds component of the time.
  void usec(long microseconds)
  {
    impl_.usec(microseconds);
  }

  /// Get the current time.
  static time now()
  {
    time tmp;
    tmp.impl_ = detail::time::now();
    return tmp;
  }

  /// Addition operator.
  void operator+=(const time& t)
  {
    impl_ += t.impl_;
  }

  /// Addition operator.
  friend time operator+(const time& a, const time& b)
  {
    time tmp(a);
    tmp.impl_ += b.impl_;
    return tmp;
  }

  /// Subtraction operator.
  void operator-=(const time& t)
  {
    impl_ -= t.impl_;
  }

  /// Subtraction operator.
  friend time operator-(const time& a, const time& b)
  {
    time tmp(a);
    tmp.impl_ -= b.impl_;
    return tmp;
  }

  /// Equality operator.
  friend bool operator==(const time& a, const time& b)
  {
    return a.impl_ == b.impl_;
  }

  /// Inequality operator.
  friend bool operator!=(const time& a, const time& b)
  {
    return !(a.impl_ == b.impl_);
  }

  /// Comparison operator.
  friend bool operator<(const time& a, const time& b)
  {
    return a.impl_ < b.impl_;
  }

  /// Comparison operator.
  friend bool operator<=(const time& a, const time& b)
  {
    return a.impl_ < b.impl_ || a.impl_ == b.impl_;
  }

  /// Comparison operator.
  friend bool operator>(const time& a, const time& b)
  {
    return !(a.impl_ < b.impl_ || a.impl_ == b.impl_);
  }

  /// Comparison operator.
  friend bool operator>=(const time& a, const time& b)
  {
    return !(a.impl_ < b.impl_);
  }

private:
  detail::time impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TIME_HPP
