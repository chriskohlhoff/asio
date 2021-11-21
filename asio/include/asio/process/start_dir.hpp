//
// process/start_dir.hpp
// ~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_PROCESS_START_DIR_HPP
#define ASIO_PROCESS_START_DIR_HPP

#include "asio/detail/config.hpp"
#include "asio/process/default_launcher.hpp"
#include "asio/detail/push_options.hpp"

namespace asio
{

struct process_start_dir
{
  filesystem::path start_dir;

  process_start_dir(filesystem::path start_dir) : start_dir(std::move(start_dir))
  {}

#if defined(ASIO_WINDOWS)
  error_code on_setup(windows::default_launcher & launcher, const filesystem::path &, const std::wstring &)
  {
    launcher.current_directory = start_dir;
    return error_code {};
  };

#else
  error_code on_exec_setup(posix::default_launcher & launcher, const filesystem::path &, const char * const *)
  {
    if (::chdir(start_dir.c_str()) == -1)
      return error_code(errno, system_category());
    else
      return error_code ();
  }
#endif

};

}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_PROCESS_START_DIR_HPP
