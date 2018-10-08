//
// io_object_impl.hpp
// ~~~~~~~~~~~~~~~~~~~
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

#include <new>
#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/io_context.hpp"
#include "asio/is_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename IoObjectService,
    typename Executor = io_context::executor_type>
class io_object_impl
{
public:
  // The type of the service that will be used to provide I/O operations.
  typedef IoObjectService service_type;

  // The underlying implementation type of I/O object.
  typedef typename service_type::implementation_type implementation_type;

  // The type of the executor associated with the object.
  typedef Executor executor_type;

  // Construct an I/O object using an executor.
  explicit io_object_impl(const executor_type& ex)
    : executor_(ex),
      service_(&asio::use_service<IoObjectService>(ex.context()))
  {
    service_->construct(implementation_);
  }

  // Construct an I/O object using an execution context.
  template <typename ExecutionContext>
  explicit io_object_impl(ExecutionContext& context,
      typename enable_if<is_convertible<
        ExecutionContext&, execution_context&>::value>::type* = 0)
    : executor_(context.get_executor()),
      service_(&asio::use_service<IoObjectService>(context))
  {
    service_->construct(implementation_);
  }

#if defined(ASIO_HAS_MOVE)
  // Move-construct an I/O object.
  io_object_impl(io_object_impl&& other)
    : executor_(other.get_executor()),
      service_(&other.get_service())
  {
    service_->move_construct(implementation_, other.implementation_);
  }

  // Perform a converting move-construction of an I/O object.
  template <typename IoObjectService1>
  io_object_impl(IoObjectService1& other_service,
      typename IoObjectService1::implementation_type& other_implementation)
    : executor_(other_service.get_io_context().get_executor()), // TODO
      service_(&asio::use_service<IoObjectService>(
            other_service.get_io_context()))
  {
    service_->converting_move_construct(implementation_,
        other_service, other_implementation);
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
    if (this != &other)
    {
      service_->move_assign(implementation_,
          *other.service_, other.implementation_);
      executor_.~executor_type();
      new (&executor_) executor_type(std::move(other.executor_));
      service_ = other.service_;
    }
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
    return executor_;
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

  // The associated executor.
  executor_type executor_;

  // The service associated with the I/O object.
  service_type* service_;

  // The underlying implementation of the I/O object.
  implementation_type implementation_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IO_OBJECT_IMPL_HPP
