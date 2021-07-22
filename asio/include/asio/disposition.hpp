//
// disposition.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DISPOSITION_HPP
#define ASIO_DISPOSITION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <concepts>
#include <exception>
#include "asio/error_code.hpp"
#include "asio/system_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename T>
struct disposition_traits;

template <>
struct disposition_traits<asio::error_code>
{
  static asio::error_code from_noerror() noexcept
  {
    return asio::error_code();
  }

  static std::exception_ptr to_exception_ptr(asio::error_code e)
  {
    return std::make_exception_ptr(asio::system_error(e));
  }
};

template <>
struct disposition_traits<std::exception_ptr>
{
  static std::exception_ptr from_noerror() noexcept
  {
    return std::exception_ptr();
  }

  static std::exception_ptr to_exception_ptr(std::exception_ptr e) noexcept
  {
    return e;
  }
};

template <typename T>
concept disposition =
  requires (T&& t)
  {
    { static_cast<bool>(std::forward<T>(t)) };
    { disposition_traits<std::decay_t<T>>::to_exception_ptr(std::forward<T>(t)) }
      -> std::same_as<std::exception_ptr>;
  };

/// The type used to singularly represent the "no error" case.
struct noerror
{
  /// Optional conversion to some error type.
  template <typename T>
  constexpr operator T() const noexcept
  {
    return disposition_traits<T>::from_noerror();
  }

  constexpr explicit operator bool() const noexcept
  {
    return false;
  }

  constexpr bool operator!() const noexcept
  {
    return true;
  }

  friend constexpr bool operator==(noerror, noerror) noexcept
  {
    return true;
  }

  friend constexpr bool operator!=(noerror, noerror) noexcept
  {
    return true;
  }

  template <typename T>
  friend constexpr bool operator==(noerror, const T& e) noexcept
  {
    return disposition_traits<T>::from_noerror() == e;
  }

  template <typename T>
  friend constexpr bool operator==(const T& e, noerror) noexcept
  {
    return disposition_traits<T>::from_noerror() == e;
  }

  template <typename T>
  friend constexpr bool operator!=(noerror, const T& e) noexcept
  {
    return disposition_traits<T>::from_noerror() != e;
  }

  template <typename T>
  friend constexpr bool operator!=(const T& e, noerror) noexcept
  {
    return disposition_traits<T>::from_noerror() != e;
  }
};

template <>
struct disposition_traits<noerror>
{
  static constexpr noerror from_noerror() noexcept
  {
    return noerror();
  }

  static std::exception_ptr to_exception_ptr(noerror) noexcept
  {
    return std::exception_ptr();
  }
};

/// A special value, similar to std::nothrow.
#if defined(ASIO_HAS_CONSTEXPR) || defined(GENERATING_DOCUMENTATION)
constexpr noerror success;
#elif defined(ASIO_MSVC)
__declspec(selectany) noerror success;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DISPOSITION_HPP
