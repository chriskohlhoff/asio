//
// select_op.hpp
// ~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_SELECT_OP_HPP
#define ASIO_DETAIL_SELECT_OP_HPP

#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class select_op_queue;

class select_op
{
public:
  // Destructor.
  virtual ~select_op();

  // Get the descriptor associated with the operation.
  socket_type descriptor() const;

  // Perform the operation.
  virtual void do_operation() = 0;

  // Handle the case where the operation has been cancelled.
  virtual void do_cancel() = 0;

protected:
  // Construct an operation for the given descriptor.
  select_op(socket_type d);

private:
  friend class select_op_queue;

  // The descriptor associated with the operation.
  socket_type descriptor_;

  // The next operation for the same file descriptor.
  select_op* next_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SELECT_OP_HPP
