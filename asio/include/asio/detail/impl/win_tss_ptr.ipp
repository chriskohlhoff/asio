//
// detail/impl/win_tss_ptr.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_WIN_TSS_PTR_IPP
#define ASIO_DETAIL_IMPL_WIN_TSS_PTR_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "../../detail/config.hpp"

#if defined(ASIO_WINDOWS)

#include "../../detail/throw_error.hpp"
#include "../../detail/win_tss_ptr.hpp"
#include "../../error.hpp"

#include "../../detail/push_options.hpp"

namespace asio {
namespace detail {

DWORD win_tss_ptr_create()
{
#if defined(UNDER_CE)
  enum { out_of_indexes = 0xFFFFFFFF };
#else
  enum { out_of_indexes = TLS_OUT_OF_INDEXES };
#endif

  DWORD tss_key = ::TlsAlloc();
  if (tss_key == out_of_indexes)
  {
    DWORD last_error = ::GetLastError();
    asio::error_code ec(last_error,
        asio::error::get_system_category());
    asio::detail::throw_error(ec, "tss");
  }
  return tss_key;
}

} // namespace detail
} // namespace asio

#include "../../detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS)

#endif // ASIO_DETAIL_IMPL_WIN_TSS_PTR_IPP
