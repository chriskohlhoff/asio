//
// impl/any_io_executor_constructors.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_ANY_IO_EXECUTOR_CONSTRUCTORS_IPP
#define ASIO_IMPL_ANY_IO_EXECUTOR_CONSTRUCTORS_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_HEADER_ONLY)
#if !defined(ASIO_USE_TS_EXECUTOR_AS_DEFAULT)

#include "asio/any_io_executor.hpp"
#include "asio/io_context.hpp"
#include "asio/strand.hpp"
#include "asio/system_executor.hpp"
#include "asio/thread_pool.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <>
any_io_executor::any_io_executor(io_context::executor_type e, int)
  : base_type(ASIO_MOVE_CAST(io_context::executor_type)(e))
{
}

template <>
any_io_executor::any_io_executor(strand<io_context::executor_type> e, int)
  : base_type(ASIO_MOVE_CAST(strand<io_context::executor_type>)(e))
{
}

template <>
any_io_executor::any_io_executor(system_executor e, int)
  : base_type(ASIO_MOVE_CAST(system_executor)(e))
{
}

template <>
any_io_executor::any_io_executor(strand<system_executor> e, int)
  : base_type(ASIO_MOVE_CAST(strand<system_executor>)(e))
{
}

template <>
any_io_executor::any_io_executor(thread_pool::executor_type e, int)
  : base_type(ASIO_MOVE_CAST(thread_pool::executor_type)(e))
{
}

template <>
any_io_executor::any_io_executor(strand<thread_pool::executor_type> e, int)
  : base_type(ASIO_MOVE_CAST(strand<thread_pool::executor_type>)(e))
{
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // !defined(ASIO_USE_TS_EXECUTOR_AS_DEFAULT)
#endif // !defined(ASIO_HEADER_ONLY)

#endif // ASIO_IMPL_ANY_IO_EXECUTOR_CONSTRUCTORS_IPP
