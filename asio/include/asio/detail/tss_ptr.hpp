//
// tss_ptr.hpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TSS_PTR_HPP
#define ASIO_DETAIL_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#if !defined(BOOST_HAS_THREADS)
# include "asio/detail/null_tss_ptr.hpp"
#elif defined(BOOST_WINDOWS)
# include "asio/detail/win_tss_ptr.hpp"
#elif defined(BOOST_HAS_PTHREADS)
# include "asio/detail/posix_tss_ptr.hpp"
#else
# error Only Windows and POSIX are supported!
#endif

namespace asio {
namespace detail {

template <typename T>
class tss_ptr
#if !defined(BOOST_HAS_THREADS)
  : public null_tss_ptr<T>
#elif defined(BOOST_WINDOWS)
  : public win_tss_ptr<T>
#elif defined(BOOST_HAS_PTHREADS)
  : public posix_tss_ptr<T>
#endif
{
public:
  void operator=(T* value)
  {
#if !defined(BOOST_HAS_THREADS)
    null_tss_ptr<T>::operator=(value);
#elif defined(BOOST_WINDOWS)
    win_tss_ptr<T>::operator=(value);
#elif defined(BOOST_HAS_PTHREADS)
    posix_tss_ptr<T>::operator=(value);
#endif
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TSS_PTR_HPP
