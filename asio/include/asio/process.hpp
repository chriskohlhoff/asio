//
// process.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_PROCESS_HPP
#define ASIO_PROCESS_HPP

#include "asio/detail/config.hpp"
#include "asio/process/default_launcher.hpp"
#include "asio/process/exit_code.hpp"
#include "asio/pid.hpp"
#include "asio/detail/push_options.hpp"

#if defined(ASIO_WINDOWS)
#include "asio/process/detail/windows_process_handle.hpp"
#else
#include "asio/process/detail/posix_process_handle.hpp"
#endif

namespace asio
{

template<typename Executor = any_io_executor>
struct basic_process
{
  using executor_type = Executor;
  executor_type get_executor() {return process_handle_.get_executor();}

  /// Provides access to underlying operating system facilities
  using native_handle_type = typename detail::basic_process_handle<executor_type>::native_handle_type;

  /** An empty process is similar to a default constructed thread. It holds an empty
  handle and is a place holder for a process that is to be launched later. */
  basic_process() = default;

  basic_process(const basic_process&) = delete;
  basic_process& operator=(const basic_process&) = delete;

  basic_process(basic_process&& lhs) : attached_(lhs.attached_), terminated_(lhs.terminated_), exit_status_{lhs.exit_status_}, process_handle_(std::move(lhs.process_handle_))
  {
    lhs.attached_ = false;
  }
  basic_process& operator=(basic_process&& lhs)
  {
    attached_ = lhs.attached_;
    terminated_ = lhs.terminated_;
    exit_status_ = lhs.exit_status_;
    process_handle_ = std::move(lhs.process_handle_);
    lhs.attached_ = false;
    return *this;
  }

  /// Construct a child from a property list and launch it.
  template<typename ... Inits>
  explicit basic_process(
      executor_type executor,
      const std::filesystem::path& exe,
      std::initializer_list<std::string_view> args,
      Inits&&... inits)
      : basic_process(default_process_launcher()(std::move(executor), exe, args, std::forward<Inits>(inits)...))
  {
  }

  /// Construct a child from a property list and launch it.
  template<typename Args, typename ... Inits>
  explicit basic_process(
      executor_type executor,
      const std::filesystem::path& exe,
      Args&& args, Inits&&... inits)
      : basic_process(default_process_launcher()(std::move(executor), exe,
                                               std::forward<Args>(args), std::forward<Inits>(inits)...))
  {
  }

  /// Construct a child from a property list and launch it.
  template<typename ExecutionContext, typename ... Inits>
  explicit basic_process(
      ExecutionContext & context,
      typename constraint<
          is_convertible<ExecutionContext&, execution_context&>::value,
          const std::filesystem::path&>::type exe,
      std::initializer_list<std::string_view> args,
      Inits&&... inits)
      : basic_process(default_process_launcher()(executor_type(context.get_executor()), exe, args, std::forward<Inits>(inits)...))
  {
  }

  /// Construct a child from a property list and launch it.
  template<typename ExecutionContext, typename Args, typename ... Inits>
  explicit basic_process(
      ExecutionContext & context,
      typename constraint<
          is_convertible<ExecutionContext&, execution_context&>::value,
          const std::filesystem::path&>::type exe,
      Args&& args, Inits&&... inits)
      : basic_process(default_process_launcher()(executor_type(context.get_executor()), exe, std::forward<Args>(args), std::forward<Inits>(inits)...))
  {
  }

  /// Attach to an existing process
  explicit basic_process(executor_type exec, pid_type pid) : process_handle_{std::move(exec), pid} {}

  /// Attach to an existing process and the internal handle
  explicit basic_process(executor_type exec, pid_type pid, native_handle_type native_handle)
        : process_handle_{std::move(exec), pid, native_handle} {}

  /// Create an invalid handle
  explicit basic_process(executor_type exec) : process_handle_{std::move(exec)} {}

  /// Attach to an existing process
  template <typename ExecutionContext>
  explicit basic_process(ExecutionContext & context, pid_type pid,
                         typename constraint<
                             is_convertible<ExecutionContext&, execution_context&>::value, void *>::type = nullptr)
       : process_handle_{context, pid} {}

  /// Attach to an existing process and the internal handle
  template <typename ExecutionContext>
  explicit basic_process(ExecutionContext & context, pid_type pid, native_handle_type native_handle,
                         typename constraint<
                            is_convertible<ExecutionContext&, execution_context&>::value, void *>::type = nullptr)
      : process_handle_{context, pid, native_handle} {}

  /// Create an invalid handle
  template <typename ExecutionContext>
  explicit basic_process(ExecutionContext & context,
                         typename constraint<
                             is_convertible<ExecutionContext&, execution_context&>::value, void *>::type = nullptr)
     : process_handle_{context} {}



  // tbd behavior
  ~basic_process()
  {
    if (attached_ && !terminated_)
      process_handle_.terminate_if_running();
  }

  void interrupt()
  {
    error_code ec;
    interrupt(ec);
    if (ec)
      throw system_error(ec, "interrupt failed");

  }
  void interrupt(error_code & ec)
  {
    process_handle_.interrupt(ec);
  }

  void request_exit()
  {
    error_code ec;
    request_exit(ec);
    if (ec)
      throw system_error(ec, "request_exit failed");
  }
  void request_exit(error_code & ec)
  {
    process_handle_.request_exit(ec);
  }

  void terminate()
  {
    error_code ec;
    terminate(ec);
    if (ec)
      detail::throw_error(ec, "terminate failed");
  }
  void terminate(error_code & ec)
  {
    process_handle_.terminate(ec, exit_status_);
  }

  void wait()
  {
    error_code ec;
    wait(ec);
    if (ec)
      detail::throw_error(ec, "wait failed");
  }
  void wait(error_code & ec)
  {
    process_handle_.wait(ec, exit_status_);
  }

  template <ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code, native_exit_code_type))
        WaitHandler ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
      ASIO_INITFN_AUTO_RESULT_TYPE(WaitHandler, void (asio::error_code, native_exit_code_type))
  async_wait(ASIO_MOVE_ARG(WaitHandler) handler ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
  {
    return process_handle_.async_wait(ASIO_MOVE_OR_LVALUE(WaitHandler)(handler));
  }

  void detach()
  {
    attached_ = false;
  }
  void join() {wait();}
  bool joinable() {return attached_; }

  native_handle_type native_handle() {return process_handle_.native_handle(); }
  int exit_code() const
  {
    return asio::evaluate_exit_code(exit_status_);
  }

  pid_type id() const {return process_handle_.id();}

  native_exit_code_type native_exit_code() const
  {
    return exit_status_;
  }
  bool running()
  {
    error_code ec;
    native_exit_code_type exit_code;
    auto r =  process_handle_.is_running(exit_code, ec);
    if (!ec)
      exit_status_ = exit_code;
    else
      throw system_error(ec, "running failed");

    return r;
  }
  bool running(error_code & ec) noexcept
  {
    native_exit_code_type exit_code ;
    auto r =  process_handle_.is_running(exit_code, ec);
    if (!ec)
      exit_status_ = exit_code;
    return r;
  }

  bool valid() const { return process_handle_.valid(); }
  explicit operator bool() const {return valid(); }

private:
  detail::basic_process_handle<Executor> process_handle_;
  bool attached_{true};
  bool terminated_{false};
  native_exit_code_type exit_status_{asio::detail::still_active};
};


typedef basic_process<> process;

}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_PROCESS_HPP
