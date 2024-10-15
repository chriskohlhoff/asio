//
// process/windows/default_launcher.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_PROCESS_WINDOWS_WITH_LOGON_LAUNCHER_HPP
#define ASIO_PROCESS_WINDOWS_WITH_LOGON_LAUNCHER_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/filesystem.hpp"
#include "asio/process/windows/as_user_launcher.hpp"

#include "asio/detail/push_options.hpp"


namespace asio
{
namespace windows
{

/// The default launcher for processes on windows.
struct as_user_launcher : default_launcher
{
  HANDLE * token;
  as_user_launcher(HANDLE * token = INVALID_HANDLE_VALUE) : token(token) {}


  template<typename ExecutionContext, typename Args, typename ... Inits>
  process operator()(ExecutionContext & context,
                     error_code & ec,
                     const typename constraint<is_convertible<
                             ExecutionContext&, execution_context&>::value,
                             filesystem::path >::type & executable,
                     Args && args,
                     Inits && ... inits )
  {
      error_code ec;
      auto proc =  (*this)(context, ec, path, std::forward<Args>(args), std::forward<Inits>(inits)...);

      if (ec)
          asio::detail::throw_error(ec, "as_user_launcher");

      return proc;
  }


  template<typename ExecutionContext, typename Args, typename ... Inits>
  auto operator()(ExecutionContext & context,
                     error_code & ec,
                     const typename constraint<is_convertible<
                             ExecutionContext&, execution_context&>::value,
                             filesystem::path >::type & executable,
                     Args && args,
                     Inits && ... inits ) -> basic_process<typename ExecutionContext::executor_type>
  {
      return (*this)(context.get_executor(), path, std::forward<Args>(args), std::forward<Inits>(inits)...);
  }

  template<typename Executor, typename Args, typename ... Inits>
  auto operator()(Executor exec,
                     error_code & ec,
                     const typename constraint<
                             execution::is_executor<Executor>::value || is_executor<Executor>::value,
                             filesystem::path >::type & executable,
                     Args && args,
                     Inits && ... inits ) -> basic_process<Executor>
  {
      error_code ec;
      auto proc =  (*this)(std::move(exec), ec, path, std::forward<Args>(args), std::forward<Inits>(inits)...);

      if (ec)
          asio::detail::throw_error(ec, "as_user_launcher");

      return proc;
  }
  }

  template<typename Executor, typename Args, typename ... Inits>
  auto operator()(Executor exec,
                     error_code & ec,
                     const typename constraint<
                             execution::is_executor<Executor>::value || is_executor<Executor>::value,
                             filesystem::path >::type & executable,
                     Args && args,
                     Inits && ... inits ) -> basic_process<Executor>
  {
    auto command_line = this->build_command_line_(executable, args);

    ec = on_init_(*this, executable, command_line, inits...);
    if (ec)
    {
      detail::on_error(*this, executable, command_line, ec, inits...);
      return basic_process<Executor>(exec);
    }
    auto ok = ::CreateProcessAsUserW(
        token,
        executable.empty() ? nullptr : executable.c_str(),
        command_line.empty() ? nullptr :  command_line.c_str(),
        process_attributes,
        thread_attributes,
        inherit_handles ? TRUE : FALSE,
        creation_flags,
        environment,
        current_directory.empty() ? nullptr : current_directory.c_str(),
        &startup_info,
        &process_information);


    if (ok == 0)
    {
      ec.assign(::GetLastError(), error::get_system_category());
      detail::on_error(*this, executable, command_line, ec, inits...);

      if (process_information.hProcess != INVALID_HANDLE)
        ::CloseHandle(process_information.hProcess);
      if (process_information.hThread != INVALID_HANDLE)
        ::CloseHandle(process_information.hThread);

      return basic_process<Executor>(exec);
    } else
    {
      detail::on_success(*this, executable, command_line, inits...);

      if (process_information.hThread != INVALID_HANDLE)
        ::CloseHandle(process_information.hThread);

      return basic_process<Executor>(exec,
                                     this->process_information.dwProcessId,
                                     this->process_information.hProcess);
    }
  }
};




}
}

#include "asio/detail/pop_options.hpp"
#endif //ASIO_PROCESS_WINDOWS_WITH_LOGON_LAUNCHER_HPP
