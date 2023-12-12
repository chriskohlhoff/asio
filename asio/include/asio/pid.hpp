//
// process/pid.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_PID_HPP
#define ASIO_PID_HPP

#include "asio/detail/config.hpp"
#include "asio/any_io_executor.hpp"

#include "asio/detail/push_options.hpp"


namespace asio
{

#if defined(GENERATING_DOCUMENTATION)

/// Implementation defined type for a process id.
typedef implementation-defined  pid_type;

#else

#if defined(ASIO_WINDOWS) \
  || defined(ASIO_WINDOWS_RUNTIME) \
  || defined(__CYGWIN__)


typedef unsigned long pid_type;

#else

typedef int pid_type;

#endif

#endif

/// Get the process id of the current process.
ASIO_DECL pid_type current_pid();


}


#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/impl/pid.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif //ASIO_PID_HPP
