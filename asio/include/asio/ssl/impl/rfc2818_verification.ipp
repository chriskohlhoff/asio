//
// ssl/impl/rfc2818_verification.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_IMPL_RFC2818_VERIFICATION_IPP
#define ASIO_SSL_IMPL_RFC2818_VERIFICATION_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <cctype>
#include <cstring>
#include "asio/ip/address.hpp"
#include "asio/ssl/rfc2818_verification.hpp"
#include "asio/ssl/detail/openssl_types.hpp"
#include <openssl/x509.h>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {

bool rfc2818_verification::operator()(
    bool preverified, verify_context& ctx) const
{
  using namespace std; // For memcmp.

  // Don't bother looking at certificates that have failed pre-verification.
  if (!preverified)
    return false;

  // We're only interested in checking the certificate at the end of the chain.
  int depth = X509_STORE_CTX_get_error_depth(ctx.native_handle());
  if (depth > 0)
    return true;

  // Try converting the host name to an address. If it is an address then we
  // need to look for an IP address in the certificate rather than a host name.
  asio::error_code ec;
  ip::address address = ip::make_address(host_, ec);
  bool const is_address = !ec;

  X509* cert = X509_STORE_CTX_get_current_cert(ctx.native_handle());

  if (is_address)
  {
  	return X509_check_ip_asc(cert, host_.c_str(), 0) == 1;
  }
  else
  {
	char* peername = nullptr;
  	int const ret = X509_check_host(cert, host_.c_str(), host_.size(), 0, &peername);
	OPENSSL_free(peername);
	return ret == 1;
  }
}

} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_IMPL_RFC2818_VERIFICATION_IPP
