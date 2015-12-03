//
// impl/await.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_AWAIT_HPP
#define ASIO_IMPL_AWAIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <exception>
#include <memory>
#include <new>
#include <tuple>
#include <utility>
#include "asio/async_result.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/dispatch.hpp"
#include "asio/post.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// Promise object for coroutine at top of thread-of-execution "stack".
class awaiter
{
public:
  awaiter* get_return_object()
  {
    return this;
  }

  auto initial_suspend()
  {
    return std::experimental::suspend_always();
  }

  auto final_suspend()
  {
    return std::experimental::suspend_always();
  }

  void set_exception(std::exception_ptr ex)
  {
    pending_exception_ = ex;
  }

  void return_void()
  {
  }

  awaiter* add_ref()
  {
    ++ref_count_;
    return this;
  }

  void release()
  {
    if (--ref_count_ == 0)
      coroutine_handle<awaiter>::from_promise(this).destroy();
  }

  void rethrow_exception()
  {
    if (pending_exception_)
    {
      std::exception_ptr ex = std::exchange(pending_exception_, nullptr);
      std::rethrow_exception(ex);
    }
  }

private:
  std::size_t ref_count_ = 0;
  std::exception_ptr pending_exception_;
};

struct awaiter_delete
{
public:
  void operator()(awaiter* a)
  {
    if (a)
      a->release();
  }
};

typedef std::unique_ptr<awaiter, awaiter_delete> awaiter_ptr;

class awaitee_base
{
public:
  auto initial_suspend()
  {
    return std::experimental::suspend_never();
  }

  struct final_suspender
  {
    awaitee_base* this_;

    bool await_ready()
    {
      return false;
    }

    void await_suspend(coroutine_handle<void>)
    {
      this_->wake_caller();
    }

    void await_resume()
    {
    }
  };

  auto final_suspend()
  {
    return final_suspender{this};
  }

  void set_exception(std::exception_ptr e)
  {
    pending_exception_ = e;
    ready_ = true;
  }

  void wake_caller()
  {
    if (caller_)
      caller_.resume();
  }

protected:
  void rethrow_exception()
  {
    if (pending_exception_)
    {
      std::exception_ptr ex = std::exchange(pending_exception_, nullptr);
      std::rethrow_exception(ex);
    }
  }

  template <typename> friend class awaitable;
  awaiter* awaiter_ = nullptr;
  coroutine_handle<void> caller_ = nullptr;
  std::exception_ptr pending_exception_ = nullptr;
  bool ready_ = false;
};

template <typename T>
class awaitee
  : public awaitee_base
{
public:
  ~awaitee()
  {
    if (initialised_)
    {
      T* p = static_cast<T*>(static_cast<void*>(buf_));
      p->~T();
    }
  }

  awaitable<T> get_return_object()
  {
    return awaitable<T>(this);
  };

  template <typename U>
  void return_value(U&& u)
  {
    T* p = static_cast<T*>(static_cast<void*>(buf_));
    new (p) T(std::forward<U>(u));
    initialised_ = true;
  }

  T value()
  {
    rethrow_exception();
    return std::move(*static_cast<T*>(static_cast<void*>(buf_)));
  }

private:
  template <typename> friend class awaitable;
  unsigned char buf_[sizeof(T)] alignas(T);
  bool initialised_ = false;
};

template <>
class awaitee<void>
  : public awaitee_base
{
public:
  awaitable<void> get_return_object()
  {
    return awaitable<void>(this);
  };

  void return_void()
  {
  }

  void value()
  {
    rethrow_exception();
  }
};

#if defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable:4033)
#endif // defined(_MSC_VER)

template <typename T> T dummy_return()
{
  return static_cast<T&&>(*static_cast<T*>(nullptr));
}

template <> void dummy_return<void>()
{
}

template <typename T>
awaitable<T> make_dummy_awaitable()
{
  for (;;) co_await std::experimental::suspend_always();
  return dummy_return<T>();
}

#if defined(_MSC_VER)
# pragma warning(pop)
#endif // defined(_MSC_VER)

template <typename Executor>
class destroy_awaiter
{
public:
  typedef Executor executor_type;

  destroy_awaiter(Executor ex, awaiter_ptr a)
    : ex_(ex),
      awaiter_(std::move(a))
  {
  }

  destroy_awaiter(destroy_awaiter&& other)
    : ex_(std::move(other.ex_)),
      awaiter_(std::move(other.awaiter_))
  {
  }

  executor_type get_executor() const noexcept
  {
    return ex_;
  }

  void operator()()
  {
    awaiter_ptr(std::move(awaiter_));
  }

protected:
  Executor ex_;
  awaiter_ptr awaiter_;
};

template <typename Executor>
class awaiter_task
{
public:
  typedef Executor executor_type;

  awaiter_task(Executor ex, awaiter* a)
    : ex_(ex),
      awaiter_(a->add_ref())
  {
  }

  awaiter_task(awaiter_task&& other)
    : ex_(std::move(other.ex_)),
      awaiter_(std::move(other.awaiter_))
  {
  }

  ~awaiter_task()
  {
    if (awaiter_)
      (post)(destroy_awaiter<Executor>(ex_, std::move(awaiter_)));
  }

  executor_type get_executor() const noexcept
  {
    return ex_;
  }

protected:
  Executor ex_;
  awaiter_ptr awaiter_;
};

template <typename Executor>
class spawn_handler : public awaiter_task<Executor>
{
public:
  using awaiter_task::awaiter_task;

  void operator()()
  {
    awaiter_ptr ptr(std::move(awaiter_));
    coroutine_handle<awaiter>::from_promise(ptr.get()).resume();
  }
};

template <typename Executor, typename T>
class await_handler_base : public awaiter_task<Executor>
{
public:
  typedef awaitable<T> awaitable_type;

  await_handler_base(basic_unsynchronized_await_context<Executor> ctx)
    : awaiter_task<Executor>(ctx.get_executor(), ctx.awaiter_),
      awaitee_(nullptr)
  {
  }

  awaitable<T> make_awaitable()
  {
    awaitable<T> a(make_dummy_awaitable<T>());
    awaitee_ = a.awaitee_;
    return a;
  }

protected:
  awaitee<T>* awaitee_;
};

template <typename Executor, typename... Args>
class await_handler;

template <typename Executor>
class await_handler<Executor>
  : public await_handler_base<Executor, void>
{
public:
  using await_handler_base::await_handler_base;

  void operator()()
  {
    awaiter_ptr ptr(std::move(awaiter_));
    awaitee_->return_void();
    awaitee_->wake_caller();
    ptr->rethrow_exception();
  }
};

template <typename Executor>
class await_handler<Executor, error_code>
  : public await_handler_base<Executor, void>
{
public:
  typedef void return_type;

  using await_handler_base::await_handler_base;

  void operator()(error_code ec)
  {
    awaiter_ptr ptr(std::move(awaiter_));
    if (ec)
      awaitee_->set_exception(std::make_exception_ptr(system_error(ec)));
    else
      awaitee_->return_void();
    awaitee_->wake_caller();
    ptr->rethrow_exception();
  }
};

template <typename Executor>
class await_handler<Executor, std::exception_ptr>
  : public await_handler_base<Executor, void>
{
public:
  using await_handler_base::await_handler_base;

  void operator()(std::exception_ptr ex)
  {
    awaiter_ptr ptr(std::move(awaiter_));
    if (ec)
      awaitee_->set_exception(ex);
    else
      awaitee_->return_void();
    awaitee_->wake_caller();
    ptr->rethrow_exception();
  }
};

template <typename Executor, typename T>
class await_handler<Executor, T>
  : public await_handler_base<Executor, T>
{
public:
  using await_handler_base::await_handler_base;

  void operator()(T t)
  {
    awaiter_ptr ptr(std::move(awaiter_));
    if (ec)
      awaitee_->set_exception(ex);
    else
      awaitee_->return_value(std::forward<T>(t));
    awaitee_->wake_caller();
    ptr->rethrow_exception();
  }
};

template <typename Executor, typename T>
class await_handler<Executor, error_code, T>
  : public await_handler_base<Executor, T>
{
public:
  using await_handler_base::await_handler_base;

  void operator()(error_code ec, T t)
  {
    awaiter_ptr ptr(std::move(awaiter_));
    if (ec)
      awaitee_->set_exception(std::make_exception_ptr(system_error(ec)));
    else
      awaitee_->return_value(std::forward<T>(t));
    awaitee_->wake_caller();
    ptr->rethrow_exception();
  }
};

template <typename Executor, typename T>
class await_handler<Executor, std::exception_ptr, T>
  : public await_handler_base<Executor, T>
{
public:
  using await_handler_base::await_handler_base;

  void operator()(std::exception_ptr ex, T t)
  {
    awaiter_ptr ptr(std::move(awaiter_));
    if (ec)
      awaitee_->set_exception(ex);
    else
      awaitee_->return_value(std::forward<T>(t));
    awaitee_->wake_caller();
    ptr->rethrow_exception();
  }
};

template <typename Executor>
class make_await_context
{
public:
  explicit make_await_context(Executor ex)
    : ex_(ex)
  {
  }

  bool await_ready()
  {
    return false;
  }

  void await_suspend(coroutine_handle<detail::awaiter> h)
  {
    awaiter_ = &h.promise();
  }

  basic_unsynchronized_await_context<Executor> await_resume()
  {
    return basic_unsynchronized_await_context<Executor>(ex_, awaiter_);
  }

private:
  Executor ex_;
  awaiter* awaiter_ = nullptr;
};

template <typename F, typename Executor, typename... Args, std::size_t... Index>
awaiter* spawn_entry_point(F f, Executor ex,
    std::tuple<Args...> args, std::index_sequence<Index...>)
{
  co_await f(co_await make_await_context<Executor>(ex),
      std::forward<Args>(std::get<Index>(args))...);
}

template <typename F, typename Executor, typename... Args>
void spawn(F f, Executor ex, Args&&... args)
{
  awaiter* a = spawn_entry_point(std::move(f), ex,
      std::forward_as_tuple(std::forward<Args>(args)...),
      std::make_index_sequence<sizeof...(Args)>());
  coroutine_handle<awaiter>::from_promise(a).resume();
  (dispatch)(spawn_handler<Executor>(ex, a));
}

} // namespace detail

template <typename T>
awaitable<T>::~awaitable()
{
  if (awaitee_)
  {
    detail::coroutine_handle<
      detail::awaitee<T>>::from_promise(
        awaitee_).destroy();
  }
}

template <typename T>
bool awaitable<T>::await_ready()
{
  return awaitee_->ready_;
}

template <typename T>
void awaitable<T>::await_suspend(detail::coroutine_handle<detail::awaiter> h)
{
  awaitee_->caller_ = h;
}

template <typename T> template <typename U>
void awaitable<T>::await_suspend(detail::coroutine_handle<detail::awaitee<U>> h)
{
  awaitee_->caller_ = h;
}

template <typename T>
T awaitable<T>::await_resume()
{
  awaitee_->caller_ = nullptr;
  return awaitee_->value();
}

template <typename Executor, typename R, typename... Args>
struct handler_type<basic_unsynchronized_await_context<Executor>, R(Args...)>
{
  typedef detail::await_handler<
    Executor, typename decay<Args>::type...> type;
};

template <typename Executor, typename... Args>
class async_result<detail::await_handler<Executor, Args...>>
{
public:
  typedef typename detail::await_handler<
    Executor, Args...>::awaitable_type type;

  async_result(detail::await_handler<Executor, Args...>& h)
    : awaitable_(h.make_awaitable())
  {
  }

  type get()
  {
    return std::move(awaitable_);
  }

private:
  type awaitable_;
};

template <typename F, typename E, typename... Args>
inline void spawn(F f, const basic_unsynchronized_await_context<E>& ctx,
    Args&&... args)
{
  detail::spawn(std::move(f), ctx.get_executor(), std::forward<Args>(args)...);
}

template <typename F, typename Executor, typename... Args>
inline auto spawn(F f, Executor ex, Args&&... args)
  -> typename enable_if<is_executor<Executor>::value>::type
{
  detail::spawn(std::move(f), ex, std::forward<Args>(args)...);
}

template <typename F, typename ExecutionContext, typename... Args>
inline auto spawn(F f, ExecutionContext& ctx, Args&&... args)
  -> typename enable_if<is_convertible<
      ExecutionContext&, execution_context&>::value>::type
{
  detail::spawn(std::move(f), ctx.get_executor(), std::forward<Args>(args)...);
}

} // namespace asio

namespace std {
namespace experimental {

template <typename F, typename Executor, typename... Args>
struct coroutine_traits<asio::detail::awaiter*, F, Executor, Args...>
{
  typedef asio::detail::awaiter promise_type;
};

template <typename T, typename... Args>
struct coroutine_traits<asio::awaitable<T>, Args...>
{
  typedef asio::detail::awaitee<T> promise_type;
};

} // namespace experimental
} // namespace std

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_AWAIT_HPP
