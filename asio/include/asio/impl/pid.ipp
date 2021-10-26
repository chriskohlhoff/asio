//
// process/impl/pid_type.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef ASIO_IMPL_PID_TYPE_IPP
#define ASIO_IMPL_PID_TYPE_IPP


#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/any_io_executor.hpp"

#include "asio/detail/push_options.hpp"

#if defined(ASIO_WINDOWS) \
  || defined(ASIO_WINDOWS_RUNTIME) \
  || defined(__CYGWIN__)

#include <windows.h>

#else

#include <unistd.h>

#endif

namespace asio
{


#if defined(ASIO_WINDOWS) \
  || defined(ASIO_WINDOWS_RUNTIME) \
  || defined(__CYGWIN__)

pid_type current_pid() {return ::GetCurrentProcessId();}

#else

pid_type current_pid() {return ::getpid();}

#endif


}


#include "asio/detail/pop_options.hpp"

#endif //ASIO_IMPL_PID_TYPE_IPP
