//
// impl/socket_base.ipp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_SOCKET_BASE_IPP
#define ASIO_IMPL_SOCKET_BASE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/socket_base.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
# include "asio/detail/apple_nw_ptr.hpp"
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
void socket_base::keep_alive::apple_nw_set(const void* self,
    nw_parameters_t parameters, nw_connection_t connection,
    asio::error_code& ec)
{
  const keep_alive* option = static_cast<const keep_alive*>(self);

  if (connection)
  {
    ec = asio::error::already_connected;
    return;
  }

  asio::detail::apple_nw_ptr<nw_protocol_stack_t> protocol_stack(
      nw_parameters_copy_default_protocol_stack(parameters));

  asio::detail::apple_nw_ptr<nw_protocol_options_t> transport_options(
      nw_protocol_stack_copy_transport_protocol(protocol_stack));

  nw_tcp_options_set_enable_keepalive(transport_options, option->value());

  ec = asio::error_code();
}

void socket_base::keep_alive::apple_nw_set(const void* self,
    nw_parameters_t parameters, nw_listener_t listener,
    asio::error_code& ec)
{
  const keep_alive* option = static_cast<const keep_alive*>(self);

  if (listener)
  {
    ec = asio::error::already_open;
    return;
  }

  asio::detail::apple_nw_ptr<nw_protocol_stack_t> protocol_stack(
      nw_parameters_copy_default_protocol_stack(parameters));

  asio::detail::apple_nw_ptr<nw_protocol_options_t> transport_options(
      nw_protocol_stack_copy_transport_protocol(protocol_stack));

  nw_tcp_options_set_enable_keepalive(transport_options, option->value());

  ec = asio::error_code();
}

void socket_base::keep_alive::apple_nw_get(void*, nw_parameters_t,
    nw_connection_t, asio::error_code& ec)
{
  ec = asio::error::operation_not_supported;
}

void socket_base::keep_alive::apple_nw_get(void*, nw_parameters_t,
    nw_listener_t, asio::error_code& ec)
{
  ec = asio::error::operation_not_supported;
}

void socket_base::reuse_address::apple_nw_set(const void* self,
    nw_parameters_t parameters, nw_connection_t connection,
    asio::error_code& ec)
{
  const reuse_address* option = static_cast<const reuse_address*>(self);

  if (connection)
  {
    ec = asio::error::already_connected;
    return;
  }

  nw_parameters_set_reuse_local_address(parameters, option->value());

  ec = asio::error_code();
}

void socket_base::reuse_address::apple_nw_set(const void* self,
    nw_parameters_t parameters, nw_listener_t listener,
    asio::error_code& ec)
{
  const reuse_address* option = static_cast<const reuse_address*>(self);

  if (listener)
  {
    ec = asio::error::already_open;
    return;
  }

  nw_parameters_set_reuse_local_address(parameters, option->value());

  ec = asio::error_code();
}

void socket_base::reuse_address::apple_nw_get(void* self,
    nw_parameters_t parameters, nw_connection_t, asio::error_code& ec)
{
  reuse_address* option = static_cast<reuse_address*>(self);

  *option = nw_parameters_get_reuse_local_address(parameters);

  ec = asio::error_code();
}

void socket_base::reuse_address::apple_nw_get(void* self,
    nw_parameters_t parameters, nw_listener_t, asio::error_code& ec)
{
  reuse_address* option = static_cast<reuse_address*>(self);

  *option = nw_parameters_get_reuse_local_address(parameters);

  ec = asio::error_code();
}
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_SOCKET_BASE_IPP
