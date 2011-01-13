//
// detail/impl/posix_event.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_POSIX_EVENT_IPP
#define ASIO_DETAIL_IMPL_POSIX_EVENT_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(BOOST_HAS_PTHREADS) && !defined(ASIO_DISABLE_THREADS)

#include "asio/detail/posix_event.hpp"
#include "asio/detail/throw_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

posix_event::posix_event()
  : signalled_(false)
{
  int error = ::pthread_cond_init(&cond_, 0);
  asio::error_code ec(error,
      asio::error::get_system_category());
  asio::detail::throw_error(ec, "event");
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(BOOST_HAS_PTHREADS) && !defined(ASIO_DISABLE_THREADS)

#endif // ASIO_DETAIL_IMPL_POSIX_EVENT_IPP
