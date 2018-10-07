//
// io_object_impl.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IO_OBJECT_IMPL_HPP
#define ASIO_DETAIL_IO_OBJECT_IMPL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/io_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename IoObjectService>
class io_object_impl
{
public:
  // The type of the service that will be used to provide I/O operations.
  typedef IoObjectService service_type;

  // The underlying implementation type of I/O object.
  typedef typename service_type::implementation_type implementation_type;

  // The type of the executor associated with the object.
  typedef asio::io_context::executor_type executor_type;

  // Construct an I/O object.
  explicit io_object_impl(asio::io_context& io_context)
    : service_(&asio::use_service<IoObjectService>(io_context))
  {
    service_->construct(implementation_);
  }

#if defined(ASIO_HAS_MOVE)
  // Move-construct an I/O object.
  io_object_impl(io_object_impl&& other)
    : service_(&other.get_service())
  {
    service_->move_construct(implementation_, other.implementation_);
  }

  // Perform a converting move-construction of an I/O object.
  template <typename IoObjectService1>
  io_object_impl(io_object_impl<IoObjectService1>&& other)
    : service_(&asio::use_service<IoObjectService>(
          other.get_service().get_io_context()))
  {
    service_->converting_move_construct(implementation_,
        other.get_service(), other.get_implementation());
  }
#endif // defined(ASIO_HAS_MOVE)

  // Destructor.
  ~io_object_impl()
  {
    service_->destroy(implementation_);
  }

#if defined(ASIO_HAS_MOVE)
  // Move-assign an I/O object.
  io_object_impl& operator=(io_object_impl&& other)
  {
    service_->move_assign(implementation_,
        *other.service_, other.implementation_);
    service_ = other.service_;
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE)

#if !defined(ASIO_NO_DEPRECATED)
  // Deprecated access to underlying I/O context.
  asio::io_context& get_io_context()
  {
    return service_->get_io_context();
  }

  // Deprecated access to underlying I/O context.
  asio::io_context& get_io_service()
  {
    return service_->get_io_context();
  }
#endif // !defined(ASIO_NO_DEPRECATED)

  // Get the executor associated with the object.
  executor_type get_executor() ASIO_NOEXCEPT
  {
    return service_->get_io_context().get_executor();
  }

  // Get the service associated with the I/O object.
  service_type& get_service()
  {
    return *service_;
  }

  // Get the service associated with the I/O object.
  const service_type& get_service() const
  {
    return *service_;
  }

  // Get the underlying implementation of the I/O object.
  implementation_type& get_implementation()
  {
    return implementation_;
  }

  // Get the underlying implementation of the I/O object.
  const implementation_type& get_implementation() const
  {
    return implementation_;
  }

private:
  // Disallow copying and copy assignment.
  io_object_impl(const io_object_impl&);
  io_object_impl& operator=(const io_object_impl&);

  // The service associated with the I/O object.
  service_type* service_;

  // The underlying implementation of the I/O object.
  implementation_type implementation_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IO_OBJECT_IMPL_HPP
