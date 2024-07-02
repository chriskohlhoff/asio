// Copyright (c) 2021 Klemens D. Morgenstern
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ASIO_PROCESS_DETAIL_POSIX_PROCESS_HANDLE_HPP
#define ASIO_PROCESS_DETAIL_POSIX_PROCESS_HANDLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)


#include "asio/detail/config.hpp"
#include "asio/posix/basic_descriptor.hpp"
#include "asio/experimental/deferred.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "asio/detail/push_options.hpp"


namespace asio
{
namespace detail
{


template<typename Executor = any_io_executor>
struct basic_process_handle
{
  typedef Executor executor_type;
  typedef posix::basic_descriptor<Executor> handle_type;
  typedef int native_handle_type;

  executor_type get_executor() {return handle_.get_executor();}

  template <ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code, int))
          WaitHandler ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
  ASIO_INITFN_AUTO_RESULT_TYPE(WaitHandler, void (asio::error_code, int))
  async_wait(ASIO_MOVE_ARG(WaitHandler) handler ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
  {
    return handle_.async_wait(posix::descriptor_base::wait_read,
            experimental::deferred(
                    [pid= pid_](std::error_code ec)
                    {
                      int exit_code{0u};
                      if (!ec)
                        if (::waitpid(pid, &exit_code, 0) == -1)
                          ec.assign(errno, error::get_system_category());

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
#if defined(SYS_pidfd_open)

  basic_process_handle(Executor executor, int pid)
          : pid_(pid), handle_(executor, syscall(SYS_pidfd_open, pid, 0))
  {
  }

#else

  basic_process_handle(Executor executor, int pid)
          : pid_(pid), handle_(executor)
  {
  }

#endif

  basic_process_handle(Executor executor, int pid, int process_handle)
          : pid_(pid), handle_(executor, process_handle)
  {
  }

  basic_process_handle(basic_process_handle &&) = default;
  basic_process_handle& operator=(basic_process_handle &&) = default;

  int native_handle() {return handle_.native_handle();}
  int id() const {return pid_;}

  void terminate_if_running()
  {
    if (handle_.native_handle() == -1)
      return ;
    if (::waitpid(pid_, nullptr, WNOHANG) == 0)
    {
      ::kill(pid_, SIGKILL);
      ::waitpid(pid_, nullptr, 0);
    }
  }

  void wait(error_code & ec, int & exit_status)
  {
    if (::waitpid(pid_, &exit_status, 0) == 1)
      ec.assign(errno, error::get_system_category());
  }

  void interrupt(error_code & ec)
  {
    if (::kill(pid_, SIGINT) == -1)
      ec.assign(errno, error::get_system_category());
  }

  void request_exit(error_code & ec)
  {
    if (::kill(pid_, SIGTERM) == -1)
      ec.assign(errno, error::get_system_category());
  }

  void terminate(error_code & ec, int & exit_status)
  {
    if (::kill(pid_, SIGKILL) == -1)
      ec.assign(errno, error::get_system_category());
  }

  bool is_running(int & exit_code, error_code ec)
  {
    if (handle_.native_handle() == -1)
    {
      ec.assign(EINVAL, error::get_system_category());
      return false;
    }
    int code = 0;
    int res = ::waitpid(pid_, &code, 0);
    if (res == -1)
      ec.assign(errno, error::get_system_category());
    else
      ec.clear();

    if (res == 0)
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
  int pid_{0};
  handle_type handle_;
};

}

}

#include "asio/detail/pop_options.hpp"


#endif //ASIO_PROCESS_DETAIL_POSIX_PROCESS_HANDLE_HPP
