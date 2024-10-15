//
// process/detail/windows/windows_process_handle.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_PROCESS_DETAIL_WINDOWS_PROCESS_HANDLE_HPP
#define ASIO_PROCESS_DETAIL_WINDOWS_PROCESS_HANDLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)


#include "asio/detail/config.hpp"
#include "asio/windows/basic_object_handle.hpp"
#include "asio/experimental/deferred.hpp"

#include <windows.h>

#include "asio/detail/push_options.hpp"

namespace asio
{
namespace detail
{

template<typename Executor = any_io_executor>
struct basic_process_handle
{
  typedef Executor executor_type;
  typedef windows::basic_object_handle<Executor> handle_type;
  typedef HANDLE native_handle_type;

  executor_type get_executor() {return handle_.get_executor();}

  template <ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code, int))
            WaitHandler ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
    ASIO_INITFN_AUTO_RESULT_TYPE(WaitHandler, void (asio::error_code, int))
  async_wait(ASIO_MOVE_ARG(WaitHandler) handler ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
  {
    return handle_.async_wait(
        experimental::deferred(
            [handle= handle_.native_handle()](std::error_code ec)
            {
              DWORD exit_code{0u};
              if (!ec)
                if (!::GetExitCodeProcess(handle, &exit_code))
                  ec.assign(::GetLastError(), error::get_system_category());

              return experimental::deferred.values(ec, exit_code);
            }))(ASIO_MOVE_OR_LVALUE(WaitHandler)(handler));
  }

  template<typename ExecutionContext>
  basic_process_handle(ExecutionContext &context,
                       typename constraint<
                           is_convertible<ExecutionContext&, execution_context&>::value
                       >::type = 0)
      : pid_(0), handle_(context)
  {
  }

  basic_process_handle(Executor executor)
      : pid_(0), handle_(executor)
  {
  }

  basic_process_handle(Executor executor, DWORD pid)
      : pid_(pid), handle_(executor, OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid))
  {
  }

  basic_process_handle(Executor executor, DWORD pid, HANDLE process_handle)
      : pid_(pid), handle_(executor, process_handle)
  {
  }

  HANDLE native_handle() {return handle_.native_handle();}
  DWORD id() const {return pid_;}

  void terminate_if_running()
  {
    DWORD exit_code = 0u;
    if (handle_.native_handle() == INVALID_HANDLE_VALUE)
      return ;
    if (::GetExitCodeProcess(handle_.native_handle(), &exit_code))
      if (exit_code == STILL_ACTIVE)
        ::TerminateProcess(handle_.native_handle(), 260);
  }

  void wait(error_code & ec, DWORD & exit_status)
  {
    if (handle_.native_handle() == INVALID_HANDLE_VALUE)
    {
      ec.assign(ERROR_INVALID_HANDLE_STATE, error::get_system_category());
      return;
    }
    handle_.wait(ec);
    if (ec)
      return;
    if (!::GetExitCodeProcess(handle_.native_handle(), &exit_status))
      ec.assign(::GetLastError(), error::get_system_category());
  }

  void interrupt(error_code & ec)
  {
    if (pid_ == 0)
    {
      ec.assign(ERROR_INVALID_HANDLE_STATE, error::get_system_category());
      return;
    }

    if (!::GenerateConsoleCtrlEvent(CTRL_C_EVENT, pid_))
      ec.assign(::GetLastError(), error::get_system_category());
  }


  struct enum_windows_data_t
  {
    error_code &ec;
    DWORD pid;
  };

  static BOOL CALLBACK enum_window(HWND hwnd, LPARAM param)
  {
    auto data = reinterpret_cast<enum_windows_data_t*>(param);
    DWORD pid{0u};
    GetWindowThreadProcessId(hwnd, &pid);

    if (pid != data->pid)
      return TRUE;

    BOOL res = ::SendMessageW(hwnd, WM_CLOSE, 0, 0);
    if (!res)
      data->ec.assign(::GetLastError(), error::get_system_category());
    return res;
  }

  void request_exit(error_code & ec)
  {
    if (pid_ == 0)
    {
      ec.assign(ERROR_INVALID_HANDLE_STATE, error::get_system_category());
      return;
    }

    enum_windows_data_t data{ec, pid_};

    if (!::EnumWindows(&basic_process_handle::enum_window, reinterpret_cast<LONG_PTR>(&data)))
      if (ec)
        ec.assign(::GetLastError(), error::get_system_category());
  }
  void terminate(error_code & ec, DWORD & exit_status)
  {
    if (handle_.native_handle() == INVALID_HANDLE_VALUE)
    {
      ec.assign(ERROR_INVALID_HANDLE_STATE, error::get_system_category());
      return;
    }
    if (!::TerminateProcess(handle_.native_handle(), 260))
      ec.assign(::GetLastError(), error::get_system_category());
    else
      wait(ec, exit_status);

  }

  bool is_running(DWORD & exit_code, error_code ec)
  {
    if (handle_.native_handle() == INVALID_HANDLE_VALUE)
    {
      ec.assign(ERROR_INVALID_HANDLE_STATE, error::get_system_category());
      return false;
    }
    DWORD code;
    //single value, not needed in the winapi.
    if (!::GetExitCodeProcess(handle_.native_handle(), &code))
      ec.assign(::GetLastError(), error::get_system_category());
    else
      ec.clear();

    if (code == still_active)
      return true;
    else
    {
      exit_code = code;
      return false;
    }
  }


  bool valid() const
  {
    return handle_.is_open();
  }


 private:
  DWORD pid_;
  handle_type handle_;
};


}
}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_PROCESS_DETAIL_WINDOWS_PROCESS_HANDLE_HPP
