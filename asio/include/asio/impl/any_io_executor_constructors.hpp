//
// impl/any_io_executor_constructors.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// No header guard

#if !defined(ASIO_HEADER_ONLY)
#if !defined(ASIO_USE_TS_EXECUTOR_AS_DEFAULT)
#if defined(ASIO_ANY_IO_EXECUTOR_DEFINED)

namespace asio {

#if defined(ASIO_IO_CONTEXT_DEFINED)

template <>
ASIO_DECL any_io_executor::any_io_executor(
    io_context::executor_type, int);

#if defined(ASIO_STRAND_DEFINED)

template <>
ASIO_DECL any_io_executor::any_io_executor(
    strand<io_context::executor_type>, int);

#endif // defined(ASIO_STRAND_DEFINED)
#endif // defined(ASIO_IO_CONTEXT_DEFINED)

#if defined(ASIO_SYSTEM_EXECUTOR_DEFINED)

template <>
ASIO_DECL any_io_executor::any_io_executor(system_executor, int);

#if defined(ASIO_STRAND_DEFINED)

template <>
ASIO_DECL any_io_executor::any_io_executor(
    strand<system_executor>, int);

#endif // defined(ASIO_STRAND_DEFINED)
#endif // defined(ASIO_SYSTEM_EXECUTOR_DEFINED)

#if defined(ASIO_THREAD_POOL_DEFINED)

template <>
ASIO_DECL any_io_executor::any_io_executor(
    thread_pool::executor_type, int);

#if defined(ASIO_STRAND_DEFINED)

template <>
ASIO_DECL any_io_executor::any_io_executor(
    strand<thread_pool::executor_type>, int);

#endif // defined(ASIO_STRAND_DEFINED)
#endif // defined(ASIO_THREAD_POOL_DEFINED)

} // namespace asio

#endif // defined(ASIO_ANY_IO_EXECUTOR_DEFINED)
#endif // !defined(ASIO_USE_TS_EXECUTOR_AS_DEFAULT)
#endif // !defined(ASIO_HEADER_ONLY)
