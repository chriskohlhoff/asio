//
// impl/error_code.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_ERROR_CODE_HPP
#define ASIO_IMPL_ERROR_CODE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

namespace asio {

inline error_code::error_code()
  : value_(0),
    category_(error::system_category)
{
}

inline error_code::error_code(error_code::value_type v, error_category c)
  : value_(v),
    category_(c)
{
}

template <typename ErrorEnum>
inline error_code::error_code(ErrorEnum e)
{
  *this = make_error_code(e);
}

/// Get the error value.
inline error_code::value_type error_code::value() const
{
  return value_;
}

/// Get the error category.
inline error_category error_code::category() const
{
  return category_;
}

inline void error_code::unspecified_bool_true(unspecified_bool_type_t)
{
}

inline error_code::operator unspecified_bool_type() const
{
  if (value_ == 0)
    return 0;
  else
    return &error_code::unspecified_bool_true;
}

inline bool error_code::operator!() const
{
  return value_ == 0;
}

inline bool operator==(const error_code& e1, const error_code& e2)
{
  return e1.value_ == e2.value_ && e1.category_ == e2.category_;
}

inline bool operator!=(const error_code& e1, const error_code& e2)
{
  return e1.value_ != e2.value_ || e1.category_ != e2.category_;
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_ERROR_CODE_HPP
