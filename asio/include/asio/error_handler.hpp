//
// error_handler.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_ERROR_HANDLER_HPP
#define ASIO_ERROR_HANDLER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {

namespace detail {

class ignore_error_t
{
public:
  typedef void result_type;

  template <typename Error>
  void operator()(const Error&) const
  {
  }
};

class throw_error_t
{
public:
  typedef void result_type;

  template <typename Error>
  void operator()(const Error& err) const
  {
    if (err)
      boost::throw_exception(err);
  }
};

template <typename Target>
class assign_error_t
{
public:
  typedef void result_type;

  assign_error_t(Target& target)
    : target_(&target)
  {
  }

  template <typename Error>
  void operator()(const Error& err) const
  {
    *target_ = err;
  }

private:
  Target* target_;
};

} // namespace detail

/**
 * @defgroup error_handler Error Handler Function Objects
 *
 * Function objects for custom error handling.
 */
/*@{*/

/// Return a function object that always ignores the error.
#if defined(GENERATING_DOCUMENTATION)
unspecified ignore_error();
#else
inline detail::ignore_error_t ignore_error()
{
  return detail::ignore_error_t();
}
#endif

/// Return a function object that always throws the error.
#if defined(GENERATING_DOCUMENTATION)
unspecified throw_error();
#else
inline detail::throw_error_t throw_error()
{
  return detail::throw_error_t();
}
#endif

/// Return a function object that assigns the error to a variable.
#if defined(GENERATING_DOCUMENTATION)
template <typename Target>
unspecified assign_error(Target& target);
#else
template <typename Target>
inline detail::assign_error_t<Target> assign_error(Target& target)
{
  return detail::assign_error_t<Target>(target);
}
#endif

/*@}*/

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ERROR_HANDLER_HPP
