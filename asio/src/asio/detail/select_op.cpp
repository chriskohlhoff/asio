//
// select_op.cpp
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

#include "asio/detail/select_op.hpp"

namespace asio {
namespace detail {

select_op::
select_op(
    socket_type d)
  : descriptor_(d),
    next_(0)
{
}

select_op::
~select_op()
{
}

socket_type
select_op::
descriptor() const
{
  return descriptor_;
}

} // namespace detail
} // namespace asio
