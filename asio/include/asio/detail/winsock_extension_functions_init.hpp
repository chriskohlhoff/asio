//
// detail/winsock_extension_functions_init.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2014 Vemund Handeland (vehandel at online dot no)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WINSOCK_EXTENSION_FUNCTIONS_INIT_HPP
#define ASIO_DETAIL_WINSOCK_EXTENSION_FUNCTIONS_INIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {


#if defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0501
  // obtains the function pointer to ConnectEx
  ASIO_DECL LPFN_CONNECTEX get_connectex(socket_type socket);

#endif // defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0501

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/winsock_extension_functions_init.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_DETAIL_WINSOCK_EXTENSION_FUNCTIONS_INIT_HPP
