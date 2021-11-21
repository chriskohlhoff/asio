//
// process/exit_code.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXIT_PROCESS_CODE_HPP
#define ASIO_EXIT_PROCESS_CODE_HPP

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS)

#else
#include <sys/wait.h>
#endif

#include "asio/detail/push_options.hpp"

namespace asio
{

#if defined(ASIO_WINDOWS)

typedef unsigned long native_exit_code_type;

namespace detail
{
constexpr native_exit_code_type still_active = 259u;
}

inline bool process_is_running(int code)
{
  return code == detail::still_active;
}

inline int evaluate_exit_code(int code)
{
  return code;
}

#else


typedef int native_exit_code_type;

namespace detail
{
constexpr native_exit_code_type still_active = 259u;
}

inline bool process_is_running(int code)
{
    return !WIFEXITED(code) && !WIFSIGNALED(code);
}

inline int evaluate_exit_code(int code)
{
  if (WIFEXITED(code))
    return WEXITSTATUS(code);
  else if (WIFSIGNALED(code))
    return WTERMSIG(code);
  else
    return code;
}

#endif

}

#endif //ASIO_EXIT_PROCESS_CODE_HPP
