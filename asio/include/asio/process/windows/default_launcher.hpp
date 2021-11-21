//
// process/windows/default_launcher.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//


#ifndef ASIO_PROCESS_WINDOWS_DEFAULT_LAUNCHER_HPP
#define ASIO_PROCESS_WINDOWS_DEFAULT_LAUNCHER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/error_code.hpp"
#include "asio/detail/filesystem.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution_context.hpp"
#include "asio/execution/executor.hpp"
#include "asio/is_executor.hpp"

#include <windows.h>

#include "asio/detail/push_options.hpp"

namespace asio
{

namespace detail
{

struct base {};
struct derived : base {};

template<typename Launcher, typename Init>
inline error_code invoke_on_setup(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line,
                                  Init && init, base && )
{
  return error_code{};
}

template<typename Launcher, typename Init>
inline auto invoke_on_setup(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line,
                            Init && init, derived && )
-> decltype(init.on_setup(launcher, executable, cmd_line))
{
  return init.on_setup(launcher, executable, cmd_line);
}

template<typename Launcher>
inline error_code on_setup(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line)
{
  return error_code{};
}

template<typename Launcher, typename Init1, typename ... Inits>
inline error_code on_setup(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line,
                           Init1 && init1, Inits && ... inits)
{
  auto ec = invoke_on_setup(launcher, executable, cmd_line, init1, derived{});
  if (ec)
    return ec;
  else
    return on_setup(launcher, executable, cmd_line, inits...);
}


template<typename Launcher, typename Init>
inline void invoke_on_error(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line,
                            const error_code & ec, Init && init, base && )
{
}

template<typename Launcher, typename Init>
inline auto invoke_on_error(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line,
                            const error_code & ec, Init && init, derived && )
-> decltype(init.on_error(launcher, ec, executable, cmd_line, ec))
{
  init.on_error(launcher, executable, cmd_line, ec);
}

template<typename Launcher>
inline void on_error(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line,
                     const error_code & ec)
{
}

template<typename Launcher, typename Init1, typename ... Inits>
inline void on_error(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line,
                     const error_code & ec,
                     Init1 && init1, Inits && ... inits)
{
  invoke_on_error(launcher, executable, cmd_line, ec, init1, derived{});
  on_error(launcher, executable, cmd_line, ec, inits...);
}

template<typename Launcher, typename Init>
inline void invoke_on_success(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line,
                                    Init && init, base && )
{
}

template<typename Launcher, typename Init>
inline auto invoke_on_success(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line,
                              Init && init, derived && )
-> decltype(init.on_success(launcher, executable, cmd_line))
{
  init.on_success(launcher, executable, cmd_line);
}

template<typename Launcher>
inline void on_success(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line)
{
}

template<typename Launcher, typename Init1, typename ... Inits>
inline void on_success(Launcher & launcher, const filesystem::path &executable, std::wstring &cmd_line,
                       Init1 && init1, Inits && ... inits)
{
  invoke_on_success(launcher, executable, cmd_line, init1, derived{});
  on_success(launcher, executable, cmd_line, inits...);
}

}

template<typename Executor>
struct basic_process;

namespace windows
{

/// The default launcher for processes on windows.
struct default_launcher
{
  SECURITY_ATTRIBUTES * process_attributes;
  SECURITY_ATTRIBUTES * thread_attributes;
  bool inherit_handles = false;
  DWORD creation_flags{EXTENDED_STARTUPINFO_PRESENT};
  void * environment = nullptr;
  filesystem::path current_directory{};

  STARTUPINFOEXW startup_info{{sizeof(STARTUPINFOEXW), nullptr, nullptr, nullptr,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nullptr,
                              INVALID_HANDLE_VALUE,
                              INVALID_HANDLE_VALUE,
                              INVALID_HANDLE_VALUE},
                              nullptr};
  PROCESS_INFORMATION process_information{nullptr, nullptr, 0,0};

  default_launcher() = default;

  template<typename ExecutionContext, typename Args, typename ... Inits>
  auto operator()(ExecutionContext & context,
                  const typename constraint<is_convertible<
                             ExecutionContext&, execution_context&>::value,
                             filesystem::path >::type & executable,
                  Args && args,
                  Inits && ... inits ) -> basic_process<typename ExecutionContext::executor_type>
  {
      error_code ec;
      auto proc =  (*this)(context, ec, executable, std::forward<Args>(args), std::forward<Inits>(inits)...);

      if (ec)
          asio::detail::throw_error(ec, "default_launcher");

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
      return (*this)(context.get_executor(), executable, std::forward<Args>(args), std::forward<Inits>(inits)...);
  }

  template<typename Executor, typename Args, typename ... Inits>
  auto operator()(Executor exec,
                  const typename constraint<
                             execution::is_executor<Executor>::value || is_executor<Executor>::value,
                             filesystem::path >::type & executable,
                  Args && args,
                  Inits && ... inits ) -> basic_process<Executor>
  {
      error_code ec;
      auto proc =  (*this)(std::move(exec), ec, executable, std::forward<Args>(args), std::forward<Inits>(inits)...);

      if (ec)
          asio::detail::throw_error(ec, "default_launcher");

      return proc;
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
    auto command_line = this->build_command_line_(executable, std::forward<Args>(args));

    ec = detail::on_setup(*this, executable, command_line, inits...);
    if (ec)
    {
      detail::on_error(*this, executable, command_line, ec, inits...);
      return basic_process<Executor>(exec);
    }
    auto ok = ::CreateProcessW(
        executable.empty() ? nullptr : executable.c_str(),
        command_line.empty() ? nullptr :  command_line.data(),
        process_attributes,
        thread_attributes,
        inherit_handles ? TRUE : FALSE,
        creation_flags,
        environment,
        current_directory.empty() ? nullptr : current_directory.c_str(),
        &startup_info.StartupInfo,
        &process_information);

    if (ok == 0)
    {
      ec.assign(::GetLastError(), error::get_system_category());
      detail::on_error(*this, executable, command_line, ec, inits...);

      if (process_information.hProcess != INVALID_HANDLE_VALUE)
        ::CloseHandle(process_information.hProcess);
      if (process_information.hThread != INVALID_HANDLE_VALUE)
        ::CloseHandle(process_information.hThread);

      return basic_process<Executor>(exec);
    }
    else
    {
      detail::on_success(*this, executable, command_line, inits...);

      if (process_information.hThread != INVALID_HANDLE_VALUE)
        ::CloseHandle(process_information.hThread);

      return basic_process<Executor>(exec,
                     this->process_information.dwProcessId,
                     this->process_information.hProcess);
    }
  }
 protected:

  template<typename Char, typename Traits = std::char_traits<char>>
  static std::wstring build_command_line_step_(ASIO_BASIC_STRING_VIEW_PARAM(Char, Traits) ws)
  {
    return build_command_line_step_(detail::convert_chars(ws.data(), ws.data() + ws.size(), L' '));
  }

  static std::wstring build_command_line_step_(std::wstring ws)
  {
    if (ws.empty())
      return L"\"\"";

    const auto has_space = ws.find(L' ') != std::wstring::npos;
    const auto quoted = (ws.front() == L'"') && (ws.back() == L'"');
    const auto needs_escape = has_space || quoted || ws.empty();
    if (!needs_escape)
      return ws;

    std::wstring res;
    res.reserve(ws.size() + 2);
    res += L'"';

    for (auto wc : ws)
    {
      if (wc == L'"')
        res += L"\"\"";
      res += wc;
    }
    res += L'"';
    return res;
  }

  template<typename Args,
           typename = decltype(build_command_line_step_(*std::begin(std::declval<Args>()))),
           typename = decltype(build_command_line_step_(*std::end(std::declval<Args>())))>
  static std::wstring build_command_line_(const filesystem::path & pt, Args && args)
  {
    std::wstring res;
    if (!pt.empty())
      res += build_command_line_step_(pt.native());

    for (auto && arg : std::forward<Args>(args))
    {
      if (!res.empty())
        res += L' ';
      res += build_command_line_step_(std::move(arg));
    }
    return res;
  }

  static std::wstring build_command_line_(const filesystem::path & pt, std::wstring ws)
  {
    if (pt.empty()) // command line version
      return ws;

    std::initializer_list<std::wstring> il={ws};
    return build_command_line_(pt, il);

  }

  template<typename Char, typename Traits>
  static std::wstring build_command_line_(const filesystem::path & pt, ASIO_BASIC_STRING_VIEW_PARAM(Char, Traits) ws)
  {
    return build_command_line_(pt, detail::convert_chars(ws.data(), ws.data() + ws.length(), L' '));
  }

  template<typename Char>
  static std::wstring build_command_line_(const filesystem::path & pt, const Char * c,
                                          typename std::char_traits<Char>::char_type * = nullptr)
  {
    return build_command_line_(pt, detail::convert_chars(c, c + std::char_traits<Char>::length(c), L' '));
  }

};


}
}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_PROCESS_WINDOWS_DEFAULT_LAUNCHER_HPP
