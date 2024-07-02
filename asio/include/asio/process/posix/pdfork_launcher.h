// Copyright (c) 2021 Klemens D. Morgenstern
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ASIO_PROCESS_POSIX_PDFORK_LAUNCHER_HPP
#define ASIO_PROCESS_POSIX_PDFORK_LAUNCHER_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/error_code.hpp"
#include "asio/detail/filesystem.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution_context.hpp"
#include "asio/execution/context.hpp"
#include "asio/execution/executor.hpp"
#include "asio/is_executor.hpp"

#include <unistd.h>
#include <sys/procdesc.h>
#include "asio/execution/context.hpp"
#incldue "asio/process/posix/default_launcher.hpp"

#include "asio/detail/push_options.hpp"

namespace asio
{
namespace posix
{

/// The default launcher for processes on windows.
struct pdfork_launcher : default_launcher
{
  int fd;

  pdfork_launcher() = default;

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
    auto argv = this->build_command_line_(executable, std::forward<Args>(args));
    {
      pipe_guard pg;
      if (::pipe(pg.p))
      {
        ec.assign(errno, system_category());
        return basic_process<Executor>{exec};
      }
      if (::fcntl(pg.p[1], F_SETFD, FD_CLOEXEC))
      {
        ec.assign(errno, system_category());
        return basic_process<Executor>{exec};
      }
      ec = detail::on_setup(*this, executable, argv, inits ...);
      if (ec)
      {
        detail::on_error(*this, executable, argv, ec, inits...);
        return basic_process<Executor>(exec);
      }

      auto & ctx = query(exec, execution::context);
      ctx.notify_fork(asio::execution_context::fork_prepare);
      pid = ::pdfork(&fd, PD_DAEMON);
      if (pid == -1)
      {
        detail::on_fork_error(*this, executable, argv, ec, inits...);
        detail::on_error(*this, executable, argv, ec, inits...);

        ec.assign(errno, system_category());
        return basic_process<Executor>{exec};
      }
      else if (pid == 0)
      {
        ctx.notify_fork(asio::execution_context::fork_child);
        ::close(pg.p[0]);

        ec = detail::on_exec_setup(*this, executable, argv, inits...);
        if (!ec)
          ::execve(executable.c_str(), const_cast<char * const *>(argv), const_cast<char * const *>(env));

        ::write(pg.p[1], &errno, sizeof(int));
        ec.assign(errno, system_category());
        detail::on_exec_error(*this, executable, argv, ec, inits...);
        ::exit(EXIT_FAILURE);
        return basic_process<Executor>{exec};
      }
      ctx.notify_fork(asio::execution_context::fork_parent);
      ::close(pg.p[1]);
      pg.p[1] = -1;
      int child_error{0};
      int count = -1;
      while ((count = ::read(pg.p[0], &child_error, sizeof(child_error))) == -1)
      {
        int err = errno;
        if ((err != EAGAIN) && (err != EINTR))
        {
          ec.assign(err, system_category());
          break;
        }
      }
      if (count != 0)
        ec.assign(child_error, system_category());

      if (ec)
      {
        detail::on_error(*this, executable, argv, ec, inits...);
        return basic_process<Executor>{exec};
      }
    }
    basic_process<Executor> proc{exec, pid, fd};
    detail::on_success(*this, executable, argv, ec, inits...);
    return proc;

  }
 protected:

  struct pipe_guard
  {
    int p[2];
    pipe_guard() : p{-1,-1} {}

    ~pipe_guard()
    {
      if (p[0] != -1)
        ::close(p[0]);
      if (p[1] != -1)
        ::close(p[1]);
    }
  };

  //if we need to allocate something
  std::vector<std::string> argv_buffer_;
  std::vector<const char *> argv_;

  template<typename Args>
  const char * const * build_command_line_(const filesystem::path & pt, Args && args,
                                           typename enable_if<
                                                   std::is_convertible<
                                                           decltype(*std::begin(std::declval<Args>())),
                                                           ASIO_CSTRING_VIEW>::value>::type * = nullptr)
  {
    const auto arg_cnt = std::distance(std::begin(args), std::end(args));
    argv_.reserve(arg_cnt + 2);
    argv_.push_back(pt.native().data());
    for (auto && arg : args)
      argv_.push_back(arg.c_str());

    argv_.push_back(nullptr);
    return argv_.data();
  }

  template<typename Args>
  const char * const *  build_command_line_(const filesystem::path & pt, Args && args,
                                            typename enable_if<
                                                    !std::is_convertible<
                                                            decltype(*std::begin(std::declval<Args>())),
                                                            ASIO_CSTRING_VIEW>::value>::type * = nullptr)
  {
    const auto arg_cnt = std::distance(std::begin(args), std::end(args));
    argv_.reserve(arg_cnt + 2);
    argv_.push_back(pt.native().data());

    using char_type = decltype(**std::begin(std::declval<Args>()));

    for (ASIO_BASIC_STRING_VIEW_PARAM(char_type) && arg : args)
      argv_buffer_.push_back(detail::convert_chars(arg.data()), arg.data() + arg.size(), ' ');

    for (auto && arg : argv_buffer_)
      argv_.push_back(arg.c_str());

    argv_.push_back(nullptr);
    return argv_.data();
  }
};


}
}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_PROCESS_POSIX_PDFORK_LAUNCHER_HPP
