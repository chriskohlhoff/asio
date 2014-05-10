//
// make_executor.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_MAKE_EXECUTOR_HPP
#define ASIO_MAKE_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/is_callable.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/unspecified_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Default hook to obtain the executor associated with a function object.
/**
 * The @c make_executor function returns the executor that will be used to
 * invoke a given function. This overload is the default used for function
 * object types that do not specify a particular executor.
 *
 * Implement @c make_executor for your own function objects and handlers to
 * customise the executor that will be used to invoke them. The Asio library
 * implementation uses argument-dependent lookup to locate the correct
 * make_executor overload as follows:
 *
 * @code using asio::make_executor;
 * auto ex = make_executor(my_function); @endcode
 *
 * For C++03, where automatic type deduction is not available, programs must
 * utilise the asio::make_executor_result type trait to determine the correct
 * executor type:
 *
 * @code typedef asio::make_executor_result<T>::type executor_type;
 * using asio::make_executor;
 * executor_type ex = make_executor(my_function); @endcode
 *
 * @note This function participates in overload resolution only when T is a
 * callable type.
 */
#if defined(GENERATING_DOCUMENTATION)
template <typename T>
unspecified_executor make_executor(const T&);
#else // defined(GENERATING_DOCUMENTATION)
template <typename T>
inline unspecified_executor make_executor(const T&,
    typename enable_if<detail::is_callable<T>::value>::type* = 0)
{
  return unspecified_executor();
}
#endif // defined(GENERATING_DOCUMENTATION)

/// Type trait used to determine associated executor type.
/**
 * Specialise @c make_executor_result for your own function objects and
 * handlers to customise the executor type that will be used to invoke them.
 */
template <typename T, typename = void>
struct make_executor_result
{
#if defined(ASIO_HAS_DECLTYPE)
private:
  static T val();
public:
  typedef decltype(make_executor((make_executor_result::val)())) type;
#else // defined(ASIO_HAS_DECLTYPE)
  typedef unspecified_executor type;
#endif // defined(ASIO_HAS_DECLTYPE)
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_MAKE_EXECUTOR_HPP
