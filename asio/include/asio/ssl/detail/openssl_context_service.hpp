//
// openssl_context_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2005 Voipster / Indrek.Juhani@voipster.com
// Copyright (c) 2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_OPENSSL_CONTEXT_SERVICE_HPP
#define ASIO_SSL_DETAIL_OPENSSL_CONTEXT_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <openssl/ssl.h>
#include "asio/detail/pop_options.hpp"

#include "asio/ssl/context_base.hpp"
#include "asio/ssl/detail/openssl_init.hpp"

namespace asio {
namespace ssl {
namespace detail {

template <typename Demuxer>
class openssl_context_service
{
public:
  // The native type of the context.
  typedef SSL_CTX* impl_type;

  // Constructor.
  openssl_context_service(Demuxer& d)
    : demuxer_(d)
  {
  }

  // The demuxer type for this service.
  typedef Demuxer demuxer_type;

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Return a null context implementation.
  static impl_type null()
  {
    return 0;
  }

  // Create a new context implementation.
  void create(impl_type& impl, context_base::method_type method)
  {
    SSL_METHOD* ssl_method = 0;
    switch (method)
    {
    case context_base::sslv2:
      ssl_method = SSLv2_method();
      break;
    case context_base::sslv2_client:
      ssl_method = SSLv2_client_method();
      break;
    case context_base::sslv2_server:
      ssl_method = SSLv2_server_method();
      break;
    case context_base::sslv3:
      ssl_method = SSLv3_method();
      break;
    case context_base::sslv3_client:
      ssl_method = SSLv3_client_method();
      break;
    case context_base::sslv3_server:
      ssl_method = SSLv3_server_method();
      break;
    case context_base::tlsv1:
      ssl_method = TLSv1_method();
      break;
    case context_base::tlsv1_client:
      ssl_method = TLSv1_client_method();
      break;
    case context_base::tlsv1_server:
      ssl_method = TLSv1_server_method();
      break;
    case context_base::sslv23:
      ssl_method = SSLv23_method();
      break;
    case context_base::sslv23_client:
      ssl_method = SSLv23_client_method();
      break;
    case context_base::sslv23_server:
      ssl_method = SSLv23_server_method();
      break;
    default:
      break;
    }
    impl = ::SSL_CTX_new(ssl_method);
  }

  // Destroy a context implementation.
  void destroy(impl_type& impl)
  {
    if (impl != null())
    {
      ::SSL_CTX_free(impl);
      impl = null();
    }
  }

private:
  // The demuxer that owns the service.
  Demuxer& demuxer_;

  // Ensure openssl is initialised.
  openssl_init<> init_;
};

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_OPENSSL_CONTEXT_SERVICE_HPP
