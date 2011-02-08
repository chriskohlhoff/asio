//
// detail/task_io_service_operation.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TASK_IO_SERVICE_OPERATION_HPP
#define ASIO_DETAIL_TASK_IO_SERVICE_OPERATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/error_code.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/task_io_service_fwd.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// Base class for all operations. A function pointer is used instead of virtual
// functions to avoid the associated overhead.
class task_io_service_operation
{
public:
  void complete(task_io_service& owner)
  {
    func_(&owner, this, asio::error_code(), 0);
  }

  void destroy()
  {
    func_(0, this, asio::error_code(), 0);
  }

protected:
  typedef void (*func_type)(task_io_service*,
      task_io_service_operation*,
      asio::error_code, std::size_t);

  task_io_service_operation(func_type func)
    : next_(0),
      func_(func)
  {
  }

  // Prevents deletion through this type.
  ~task_io_service_operation()
  {
  }

private:
  friend class op_queue_access;
  task_io_service_operation* next_;
  func_type func_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TASK_IO_SERVICE_OPERATION_HPP
