//
// service_unavailable.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_SERVICE_UNAVAILABLE_HPP
#define ASIO_SERVICE_UNAVAILABLE_HPP

#include <stdexcept>
#include "asio/service_type_id.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// The service_unavailable exception may be thrown when the demuxer is asked
/// to supply a service interface that is not supported.
class service_unavailable
  : public std::runtime_error
{
public:
  /// Constructor.
  service_unavailable(const service_type_id& type);

  /// Get the type of the unavailable service.
  const service_type_id& service_type() const;

private:
  /// The type of the unavailable service.
  const service_type_id service_type_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SERVICE_UNAVAILABLE_HPP
