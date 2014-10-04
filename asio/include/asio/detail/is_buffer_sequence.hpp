//
// detail/is_buffer_sequence.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct begin_end_memfns_base
{
  void begin();
  void end();
};

template <typename T>
struct begin_end_memfns_derived
  : T, begin_end_memfns_base
{
};

template <typename T, T>
struct begin_end_memfns_check
{
};

template <typename>
char (&begin_memfn_helper(...))[2];

template <typename T>
char begin_memfn_helper(
    begin_end_memfns_check<
      void (begin_end_memfns_base::*)(),
      &begin_end_memfns_derived<T>::begin>*);

template <typename>
char (&end_memfn_helper(...))[2];

template <typename T>
char end_memfn_helper(
    begin_end_memfns_check<
      void (begin_end_memfns_base::*)(),
      &begin_end_memfns_derived<T>::end>*);

template <typename, typename>
char (&value_type_const_iterator_typedefs_helper(...))[2];

template <typename T, typename Buffer>
char value_type_const_iterator_typedefs_helper(
    typename T::const_iterator*,
    typename enable_if<is_convertible<
      typename T::value_type, Buffer>::value>::type*);

template <typename T, typename Buffer>
struct is_buffer_sequence_class
  : integral_constant<bool,
      sizeof(begin_memfn_helper<T>(0)) != 1 &&
      sizeof(end_memfn_helper<T>(0)) != 1 &&
      sizeof(value_type_const_iterator_typedefs_helper<T, Buffer>(0, 0)) == 1>
{
};

template <typename T, typename Buffer>
struct is_buffer_sequence
  : conditional<is_class<T>::value,
      is_buffer_sequence_class<T, Buffer>,
      false_type>::type
{
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IS_BUFFER_SEQUENCE_HPP
