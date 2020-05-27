//
// security/on_certificate_verify.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SECURITY_ON_CERTIFICATE_VERIFY_HPP
#define ASIO_SECURITY_ON_CERTIFICATE_VERIFY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/detail/apple_nw_ptr.hpp"
#include <Network/Network.h>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace security {

/// Socket option to set the certificate verification callback.
template <typename Function>
class on_certificate_verify
{
public:
  explicit on_certificate_verify(Function f)
    : f_(ASIO_MOVE_CAST(Function)(f))
  {
  }

  // The following functions comprise the extensible interface for the
  // SettableSocketOption concept when targeting the Apple Network Framework.

  // Set the socket option on the specified connection.
  static void apple_nw_set(const void* self, nw_parameters_t parameters,
      nw_connection_t, asio::error_code& ec)
  {
    static_cast<const on_certificate_verify*>(self)->do_set(parameters, ec);
  }

  // Set the socket option on the specified listener.
  static void apple_nw_set(const void* self,
      nw_parameters_t parameters, nw_listener_t,
      asio::error_code& ec)
  {
    static_cast<const on_certificate_verify*>(self)->do_set(parameters, ec);
  }

private:
  void do_set(nw_parameters_t parameters, asio::error_code& ec) const
  {
    asio::detail::apple_nw_ptr<nw_protocol_stack_t> protocol_stack(
        nw_parameters_copy_default_protocol_stack(parameters));

    asio::detail::apple_nw_ptr<nw_protocol_definition_t> tls_definition(
        nw_protocol_copy_tls_definition());

    nw_protocol_stack_iterate_application_protocols(protocol_stack,
        ^(nw_protocol_options_t protocol)
        {
          asio::detail::apple_nw_ptr<nw_protocol_definition_t> definition(
              nw_protocol_options_copy_definition(protocol));

          if (nw_protocol_definition_is_equal(definition, tls_definition))
          {
            asio::detail::apple_nw_ptr<sec_protocol_options_t> sec_options(
                nw_tls_copy_sec_protocol_options(protocol));

            __block Function f(f_);
            sec_protocol_options_set_verify_block(sec_options,
                Block_copy(
                  ^(sec_protocol_metadata_t metadata,
                  sec_trust_t trust_ref,
                  sec_protocol_verify_complete_t complete)
                {
                  f(metadata, trust_ref, complete);
                }),
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));
          }
        });

    ec = asio::error_code();
  }

  Function f_;
};

} // namespace security
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#endif // ASIO_SECURITY_ON_CERTIFICATE_VERIFY_HPP
