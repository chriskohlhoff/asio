//
// detail/impl/winsock_extension_functions_init.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2014 Vemund Handeland (vehandel at online dot no)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_WINSOCK_EXTENSION_FUNCTIONS_INIT_IPP
#define ASIO_DETAIL_IMPL_WINSOCK_EXTENSION_FUNCTIONS_INIT_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/winsock_extension_functions_init.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/detail/winsock_extension_functions_init.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {


#if defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0501

  LPFN_CONNECTEX get_connectex(socket_type socket)
  {
    static LPFN_CONNECTEX connectex_func;

    // Kind of lazy initialization, but does not prevent multiple threads from actually initializing the pointer.
    if (connectex_func == 0)
    {
      // Try to get the ConnectEx function pointer.
      LPFN_CONNECTEX func_ptr = 0;

      GUID connectex_guid = WSAID_CONNECTEX;
      DWORD ignored_bytes_returned;
      if (::WSAIoctl(socket, SIO_GET_EXTENSION_FUNCTION_POINTER,
        &connectex_guid, sizeof(connectex_guid),
        &func_ptr, sizeof(func_ptr),
        &ignored_bytes_returned, 0, 0) == 0)
      {
        // Force immediate write to main memory.
        ::InterlockedExchangePointer(reinterpret_cast<void**>(&connectex_func), func_ptr);
      }
    }

    return connectex_func;
  }

#endif // defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0501

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_IMPL_WINSOCK_EXTENSION_FUNCTIONS_INIT_IPP
