//
// detail/is_buffer_sequence.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2019 Alexander Karzhenkov
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IS_BUFFER_SEQUENCE_HPP
#define ASIO_DETAIL_IS_BUFFER_SEQUENCE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/sfinae_helpers.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

class mutable_buffer;
class const_buffer;

namespace detail {

template <typename Buffer>
struct is_buffer_sequence_class : sfinae_check_base
{
  template <typename T>
  static result<sizeof(

    // Basic checks

    is_convertible_to<Buffer>(*buffer_sequence_begin(declval<T&>())),
    is_convertible_to<Buffer>(*buffer_sequence_end(declval<T&>())),

#if 1

    // Check additional details of the inspected type

    // Perhaps the additional checks make sense when using C++98.
    // It would be better to implement them by the means
    // of 'buffer_sequence_begin' and 'buffer_sequence_end'.
    // Explicit specializations of 'is_buffer_sequence'
    // for 'mutable_buffer' and 'const_buffer' would not be needed.

    is_convertible_to<Buffer>(*declval<T>().begin()),
    is_convertible_to<Buffer>(*declval<T>().end()),

    is_convertible_to<Buffer>(declval<typename T::value_type>()),

    check<typename T::const_iterator>(),

#endif

  0)> detector(int);
};

template <typename T, typename Buffer>
struct is_buffer_sequence
  : sfinae_result<is_buffer_sequence_class<Buffer>, T>
{
};

template <>
struct is_buffer_sequence<mutable_buffer, mutable_buffer>
  : true_type
{
};

template <>
struct is_buffer_sequence<mutable_buffer, const_buffer>
  : true_type
{
};

template <>
struct is_buffer_sequence<const_buffer, const_buffer>
  : true_type
{
};

template <>
struct is_buffer_sequence<const_buffer, mutable_buffer>
  : false_type
{
};

struct is_dynamic_buffer_class_v1 : sfinae_check_base
{
  static const std::size_t arg = 0;

  template <typename T>
  static result<sizeof(

    is_same_as<std::size_t>(declval<T>().size()),
    is_same_as<std::size_t>(declval<T>().max_size()),
    is_same_as<std::size_t>(declval<T>().capacity()),

    declval<T>().consume(arg),
    declval<T>().commit(arg),

    is_same_as<typename T::const_buffers_type>(
      declval<T>().data()),

    is_same_as<typename T::mutable_buffers_type>(
      declval<T>().prepare(arg)),

    check<is_buffer_sequence<
      typename T::const_buffers_type, const_buffer>::value>(),

    check<is_buffer_sequence<
      typename T::mutable_buffers_type, mutable_buffer>::value>(),

  0)> detector(int);
};

template <typename T>
struct is_dynamic_buffer_v1
  : sfinae_result<is_dynamic_buffer_class_v1,
      typename remove_const<T>::type>
{
};

struct is_dynamic_buffer_class_v2 : sfinae_check_base
{
  static const std::size_t arg = 0;

  template <typename T>
  static result<sizeof(

    is_same_as<std::size_t>(declval<T>().size()),
    is_same_as<std::size_t>(declval<T>().max_size()),
    is_same_as<std::size_t>(declval<T>().capacity()),

    declval<T>().consume(arg),
    declval<T>().grow(arg),
    declval<T>().shrink(arg),

    is_same_as<typename T::mutable_buffers_type>(
      declval<T>().data(arg, arg)),

    is_same_as<typename T::const_buffers_type>(
      declval<const T>().data(arg, arg)),

    check<is_buffer_sequence<
      typename T::const_buffers_type, const_buffer>::value>(),

    check<is_buffer_sequence<
      typename T::mutable_buffers_type, mutable_buffer>::value>(),

  0)> detector(int);
};

template <typename T>
struct is_dynamic_buffer_v2
  : sfinae_result<is_dynamic_buffer_class_v2,
      typename remove_const<T>::type>
{
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IS_BUFFER_SEQUENCE_HPP
