//
// service_type_id.hpp
// ~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_SERVICE_TYPE_ID_HPP
#define ASIO_SERVICE_TYPE_ID_HPP

#include "asio/detail/push_options.hpp"

namespace asio {

/// The service_type_id class implements an automatic unique identifier.
class service_type_id
{
public:
  /// Default constructor, creates a unique id.
  service_type_id() : unique_id_(&unique_id_) {}

  /// Compare two ids for equality.
  friend bool operator==(const service_type_id& id1, const service_type_id& id2)
  {
    return id1.unique_id_ == id2.unique_id_;
  }

  /// Compare two ids for inequality.
  friend bool operator!=(const service_type_id& id1, const service_type_id& id2)
  {
    return id1.unique_id_ != id2.unique_id_;
  }

  /// Less-than operator for two ids.
  friend bool operator<(const service_type_id& id1, const service_type_id& id2)
  {
    return id1.unique_id_ < id2.unique_id_;
  }

  /// Greater-than operator for two ids.
  friend bool operator>(const service_type_id& id1, const service_type_id& id2)
  {
    return id1.unique_id_ > id2.unique_id_;
  }

private:
  /// The address of a void* pointer is used to provide a unique id.
  void* unique_id_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SERVICE_TYPE_ID_HPP
