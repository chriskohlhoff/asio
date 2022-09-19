//
// impl/use_awaitable.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2022 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_USE_AWAITABLE_HPP
#define ASIO_IMPL_USE_AWAITABLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/async_result.hpp"
#include "asio/cancellation_signal.hpp"
#include "asio/early_complete.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Executor, typename T>
class awaitable_handler_base
  : public awaitable_thread<Executor>
{
public:
  typedef void result_type;
  typedef awaitable<T, Executor> awaitable_type;

  // Construct from the entry point of a new thread of execution.
  awaitable_handler_base(awaitable<awaitable_thread_entry_point, Executor> a,
      const Executor& ex, cancellation_slot pcs, cancellation_state cs)
    : awaitable_thread<Executor>(std::move(a), ex, pcs, cs)
  {
  }

  // Transfer ownership from another awaitable_thread.
  explicit awaitable_handler_base(awaitable_thread<Executor>* h)
    : awaitable_thread<Executor>(std::move(*h))
  {
  }

protected:
  awaitable_frame<T, Executor>* frame() noexcept
  {
    return static_cast<awaitable_frame<T, Executor>*>(
        this->entry_point()->top_of_stack_);
  }
};

template <typename, typename...>
class awaitable_handler;

template <typename Executor>
class awaitable_handler<Executor>
  : public awaitable_handler_base<Executor, void>
{
public:
  using awaitable_handler_base<Executor, void>::awaitable_handler_base;

  void operator()()
  {
    this->frame()->attach_thread(this);
    this->frame()->return_void();
    this->frame()->clear_cancellation_slot();
    this->frame()->pop_frame();
    this->pump();
  }
};

template <typename Executor>
class awaitable_handler<Executor, asio::error_code>
  : public awaitable_handler_base<Executor, void>
{
public:
  using awaitable_handler_base<Executor, void>::awaitable_handler_base;

  void operator()(const asio::error_code& ec)
  {
    this->frame()->attach_thread(this);
    if (ec)
      this->frame()->set_error(ec);
    else
      this->frame()->return_void();
    this->frame()->clear_cancellation_slot();
    this->frame()->pop_frame();
    this->pump();
  }
};

template <typename Executor>
class awaitable_handler<Executor, std::exception_ptr>
  : public awaitable_handler_base<Executor, void>
{
public:
  using awaitable_handler_base<Executor, void>::awaitable_handler_base;

  void operator()(std::exception_ptr ex)
  {
    this->frame()->attach_thread(this);
    if (ex)
      this->frame()->set_except(ex);
    else
      this->frame()->return_void();
    this->frame()->clear_cancellation_slot();
    this->frame()->pop_frame();
    this->pump();
  }
};

template <typename Executor, typename T>
class awaitable_handler<Executor, T>
  : public awaitable_handler_base<Executor, T>
{
public:
  using awaitable_handler_base<Executor, T>::awaitable_handler_base;

  template <typename Arg>
  void operator()(Arg&& arg)
  {
    this->frame()->attach_thread(this);
    this->frame()->return_value(std::forward<Arg>(arg));
    this->frame()->clear_cancellation_slot();
    this->frame()->pop_frame();
    this->pump();
  }
};

template <typename Executor, typename T>
class awaitable_handler<Executor, asio::error_code, T>
  : public awaitable_handler_base<Executor, T>
{
public:
  using awaitable_handler_base<Executor, T>::awaitable_handler_base;

  template <typename Arg>
  void operator()(const asio::error_code& ec, Arg&& arg)
  {
    this->frame()->attach_thread(this);
    if (ec)
      this->frame()->set_error(ec);
    else
      this->frame()->return_value(std::forward<Arg>(arg));
    this->frame()->clear_cancellation_slot();
    this->frame()->pop_frame();
    this->pump();
  }
};

template <typename Executor, typename T>
class awaitable_handler<Executor, std::exception_ptr, T>
  : public awaitable_handler_base<Executor, T>
{
public:
  using awaitable_handler_base<Executor, T>::awaitable_handler_base;

  template <typename Arg>
  void operator()(std::exception_ptr ex, Arg&& arg)
  {
    this->frame()->attach_thread(this);
    if (ex)
      this->frame()->set_except(ex);
    else
      this->frame()->return_value(std::forward<Arg>(arg));
    this->frame()->clear_cancellation_slot();
    this->frame()->pop_frame();
    this->pump();
  }
};

template <typename Executor, typename... Ts>
class awaitable_handler
  : public awaitable_handler_base<Executor, std::tuple<Ts...>>
{
public:
  using awaitable_handler_base<Executor,
    std::tuple<Ts...>>::awaitable_handler_base;

  template <typename... Args>
  void operator()(Args&&... args)
  {
    this->frame()->attach_thread(this);
    this->frame()->return_values(std::forward<Args>(args)...);
    this->frame()->clear_cancellation_slot();
    this->frame()->pop_frame();
    this->pump();
  }
};

template <typename Executor, typename... Ts>
class awaitable_handler<Executor, asio::error_code, Ts...>
  : public awaitable_handler_base<Executor, std::tuple<Ts...>>
{
public:
  using awaitable_handler_base<Executor,
    std::tuple<Ts...>>::awaitable_handler_base;

  template <typename... Args>
  void operator()(const asio::error_code& ec, Args&&... args)
  {
    this->frame()->attach_thread(this);
    if (ec)
      this->frame()->set_error(ec);
    else
      this->frame()->return_values(std::forward<Args>(args)...);
    this->frame()->clear_cancellation_slot();
    this->frame()->pop_frame();
    this->pump();
  }
};

template <typename Executor, typename... Ts>
class awaitable_handler<Executor, std::exception_ptr, Ts...>
  : public awaitable_handler_base<Executor, std::tuple<Ts...>>
{
public:
  using awaitable_handler_base<Executor,
    std::tuple<Ts...>>::awaitable_handler_base;

  template <typename... Args>
  void operator()(std::exception_ptr ex, Args&&... args)
  {
    this->frame()->attach_thread(this);
    if (ex)
      this->frame()->set_except(ex);
    else
      this->frame()->return_values(std::forward<Args>(args)...);
    this->frame()->clear_cancellation_slot();
    this->frame()->pop_frame();
    this->pump();
  }
};

template<typename ... Ts>
struct result_helper
{
  std::optional<std::tuple<Ts... >> res;
  void set(Ts && ... ts)
  {
    res.emplace(std::move(ts)...);
  }

  std::tuple<Ts... > get()
  {
    return std::move(res.get());
  }
};


template<>
struct result_helper<>
{
  void set() { }
  void get() { }
};

template<>
struct result_helper<error_code>
{
  error_code ec;
  void set(error_code ec_)
  {
    ec = ec_;
  }

  void get()
  {
    if (ec)
      detail::throw_error(ec);
  }
};


template<>
struct result_helper<std::exception_ptr>
{
  std::exception_ptr ex;
  void set(std::exception_ptr ex_)
  {
    ex = ex_;
  }

  void get()
  {
    if (ex)
      std::rethrow_exception(ex);
  }
};

template<typename T>
struct result_helper<error_code, T>
{
  std::optional<T> res;
  error_code ec;
  void set(error_code ec_, T value)
  {
    ec = ec_;
    res.emplace(std::move(value));
  }

  T get()
  {
    if (ec)
      detail::throw_error(ec);
    return std::move(*res);
  }
};


template<typename T>
struct result_helper<std::exception_ptr, T>
{
  std::optional<T> res;
  std::exception_ptr ex;
  void set(std::exception_ptr ex_, T value)
  {
    ex = ex_;
    res.emplace(std::move(value));
  }

  T get()
  {
    if (ex)
      std::rethrow_exception(ex);
    return std::move(*res);
  }
};


template<typename ... Ts>
struct result_helper<error_code, Ts...>
{
  std::optional<std::tuple<Ts...>> res;
  error_code ec;
  void set(error_code ec_, Ts ...  value)
  {
    ec = ec_;
    res.emplace(std::move(value)...);
  }

  std::tuple<Ts...> get()
  {
    if (ec)
      detail::throw_error(ec);
    return std::move(*res);
  }
};


template<typename ... Ts>
struct result_helper<std::exception_ptr, Ts...>
{
  std::optional<std::tuple<Ts...>> res;
  std::exception_ptr ex;
  void set(std::exception_ptr ex_, Ts ...  value)
  {
    ex = ex_;
    res.emplace(std::move(value)...);
  }

  std::tuple<Ts...> get()
  {
    if (ex)
      std::rethrow_exception(ex);
    return std::move(*res);
  }
};




} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

#if defined(_MSC_VER)
template <typename T>
T dummy_return()
{
  return std::move(*static_cast<T*>(nullptr));
}

template <>
inline void dummy_return()
{
}
#endif // defined(_MSC_VER)

template <typename Executor, typename R, typename... Args>
class async_result<use_awaitable_t<Executor>, R(Args...)>
{
public:
  typedef typename detail::awaitable_handler<
      Executor, typename decay<Args>::type...> handler_type;
  typedef typename handler_type::awaitable_type return_type;

  template <typename Initiation, typename... InitArgs>
#if defined(__APPLE_CC__) && (__clang_major__ == 13)
  __attribute__((noinline))
#endif // defined(__APPLE_CC__) && (__clang_major__ == 13)
  static handler_type* do_init(
      detail::awaitable_frame_base<Executor>* frame, Initiation& initiation,
      use_awaitable_t<Executor> u, InitArgs&... args)
  {
    (void)u;
    ASIO_HANDLER_LOCATION((u.file_name_, u.line_, u.function_name_));
    handler_type handler(frame->detach_thread());
    std::move(initiation)(std::move(handler), std::move(args)...);
    return nullptr;
  }

  template <typename Initiation, typename... InitArgs>
  static return_type initiate(Initiation initiation,
      use_awaitable_t<Executor> u, InitArgs... args)
  {
    early_completion_helper<Initiation,
            use_awaitable_t<Executor>,
            R(Args...)> ech(initiation);

    if (ech.try_complete(ASIO_MOVE_CAST(InitArgs)(args)...))
    {
      detail::result_helper<Args...> res;
      ech.receive([&](Args ... args_){res.set(std::move(args_)...);});
      co_return res.get();
    }

    co_await [&] (auto* frame)
      {
        return do_init(frame, initiation, u, args...);
      };

    for (;;) {} // Never reached.
#if defined(_MSC_VER)
    co_return dummy_return<typename return_type::value_type>();
#endif // defined(_MSC_VER)
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_USE_AWAITABLE_HPP
