//
// timer_base.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_TIMER_BASE_HPP
#define ASIO_TIMER_BASE_HPP

#include "asio/detail/push_options.hpp"

namespace asio {

/// The timer_base class is used as a base for the basic_timer class template
/// so that we have a common place to define the from_type enum.
class timer_base
{
public:
  /// The point from where relative times are measured.
  enum from_type
  {
    /// The specified time is relative to the current time.
    from_now,

    /// The specified time is relative to the timer's existing expiry value.
    from_existing,

    /// The specified time is a "C" time, i.e. it is measured from midnight,
    /// January 1, 1970.
    from_epoch
  };

protected:
  /// Protected destructor to prevent deletion through this type.
  ~timer_base()
  {
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TIMER_BASE_HPP
