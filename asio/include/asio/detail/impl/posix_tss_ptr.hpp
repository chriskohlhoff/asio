//
// posix_tss_ptr.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_POSIX_TSS_PTR_HPP
#define ASIO_DETAIL_IMPL_POSIX_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/posix_tss_ptr.hpp"

#if defined(BOOST_HAS_PTHREADS)

namespace asio {
namespace detail {

template <typename T>
inline posix_tss_ptr<T>::posix_tss_ptr()
{
  posix_tss_ptr_create(tss_key_);
}

template <typename T>
inline posix_tss_ptr<T>::~posix_tss_ptr()
{
  ::pthread_key_delete(tss_key_);
}

template <typename T>
inline posix_tss_ptr<T>::operator T*() const
{
  return static_cast<T*>(::pthread_getspecific(tss_key_));
}

template <typename T>
inline void posix_tss_ptr<T>::operator=(T* value)
{
  ::pthread_setspecific(tss_key_, value);
}

} // namespace detail
} // namespace asio

#endif // defined(BOOST_HAS_PTHREADS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_POSIX_TSS_PTR_HPP
