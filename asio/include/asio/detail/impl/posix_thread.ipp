//
// detail/impl/posix_thread.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_POSIX_THREAD_IPP
#define ASIO_DETAIL_IMPL_POSIX_THREAD_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/posix_thread.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#if defined(BOOST_HAS_PTHREADS)

#include "asio/error_code.hpp"
#include "asio/detail/throw_error.hpp"

namespace asio {
namespace detail {

posix_thread::~posix_thread()
{
  if (!joined_)
    ::pthread_detach(thread_);
}

void posix_thread::join()
{
  if (!joined_)
  {
    ::pthread_join(thread_, 0);
    joined_ = true;
  }
}

void posix_thread::start_thread(func_base* arg)
{
  int error = ::pthread_create(&thread_, 0,
        asio_detail_posix_thread_function, arg);
  if (error != 0)
  {
    delete arg;
    asio::system_error e(
        asio::error_code(error,
          asio::error::get_system_category()),
        "thread");
    boost::throw_exception(e);
  }
}

void* asio_detail_posix_thread_function(void* arg)
{
  posix_thread::auto_func_base_ptr func = {
      static_cast<posix_thread::func_base*>(arg) };
  func.ptr->run();
  return 0;
}

} // namespace detail
} // namespace asio

#endif // defined(BOOST_HAS_PTHREADS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_POSIX_THREAD_IPP
