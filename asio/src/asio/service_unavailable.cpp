//
// service_unavailable.cpp
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

#include "asio/service_unavailable.hpp"

namespace asio {

service_unavailable::
service_unavailable(
    const service_type_id& type)
  : std::runtime_error("Service unavailable"),
    service_type_(type)
{
}

const service_type_id&
service_unavailable::
service_type() const
{
  return service_type_;
}

} // namespace asio
