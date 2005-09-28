//
// basic_context.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2005 Voipster / Indrek.Juhani@voipster.com
// Copyright (c) 2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_BASIC_CONTEXT_HPP
#define ASIO_SSL_BASIC_CONTEXT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/service_factory.hpp"
#include "asio/ssl/context_base.hpp"

namespace asio {
namespace ssl {

/// SSL context.
template <typename Service>
class basic_context
  : public context_base,
    private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide context operations.
  typedef Service service_type;

  /// The native implementation type of the locking dispatcher.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this context.
  typedef typename service_type::demuxer_type demuxer_type;

  /// Constructor.
  basic_context(demuxer_type& d, method_type method)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_, method);
  }

  /// Destructor.
  ~basic_context()
  {
    service_.destroy(impl_);
  }

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * context. This is intended to allow access to context functionality that is
   * not otherwise provided.
   */
  impl_type impl()
  {
    return impl_;
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_BASIC_CONTEXT_HPP
