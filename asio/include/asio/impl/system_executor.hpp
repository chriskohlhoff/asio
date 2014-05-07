//
// impl/system_executor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_SYSTEM_EXECUTOR_HPP
#define ASIO_IMPL_SYSTEM_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/assert.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename Function>
void system_executor::dispatch(ASIO_MOVE_ARG(Function) f)
{
  typename decay<Function>::type tmp(ASIO_MOVE_CAST(Function)(f));
  tmp();
}

template <typename Function>
void system_executor::post(ASIO_MOVE_ARG(Function) f)
{
  ASIO_ASSERT(0 && "Not yet implemented");
  (void)f;
}

template <typename Function>
void system_executor::defer(ASIO_MOVE_ARG(Function) f)
{
  ASIO_ASSERT(0 && "Not yet implemented");
  (void)f;
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_SYSTEM_EXECUTOR_HPP
