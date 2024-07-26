//
// error_traits.hpp
// ~~~~~~~~~~~~~~~~
//
//
// Copyright (c) 2024 Klemens Morgenstern (klemens.morgenstern@gmx.net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_ERROR_TRAITS_HPP
#define ASIO_ERROR_TRAITS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Error traits to automatically convert error values into exceptions.
template<typename Error>
struct error_traits;

template<>
struct error_traits<asio::error_code>
{
  typedef asio::error_code type;

  ASIO_NORETURN
  static void throw_error(
      const asio::error_code& err
      ASIO_SOURCE_LOCATION_PARAM)
  {
    detail::do_throw_error(err ASIO_SOURCE_LOCATION_ARG);
  }

  ASIO_NODISCARD
  static bool is_failure(
      const asio::error_code  &ec) ASIO_NOEXCEPT
  {
     return !!ec;
  }
};


template<>
struct error_traits<std::exception_ptr>
{
  typedef std::exception_ptr type;

  ASIO_NORETURN
  static void throw_error(
      std::exception_ptr e)
  {
    std::rethrow_exception(e);
  }

  ASIO_NODISCARD
  static bool is_failure(
      const std::exception_ptr &e) ASIO_NOEXCEPT
  {
    return !!e;
  }
};

template<typename Error, typename = Error>
struct is_error : std::false_type {};

template<typename Error>
struct is_error<Error, typename error_traits<Error>::type> : std::true_type {};

template<>
struct is_error<std::exception_ptr, std::exception_ptr> : std::true_type {};

template<>
struct is_error<asio::error_code, asio::error_code> : std::true_type {};

template<typename T>
struct signature_has_error : std::false_type {};

template<>
struct signature_has_error<void()> : std::false_type {};

template<typename T>
struct signature_has_error<void(T)> : is_error<T> {};


template<typename T, typename ... Ts>
struct signature_has_error<void(T, Ts...)> : is_error<T> {};


}

#include "asio/detail/pop_options.hpp"


#endif //ASIO_ERROR_TRAITS_HPP
