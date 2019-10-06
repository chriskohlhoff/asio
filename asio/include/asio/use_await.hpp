//
// use_await.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_USE_AWAIT_HPP
#define ASIO_USE_AWAIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_CO_AWAIT) || defined(GENERATING_DOCUMENTATION)

#include <functional>
#include <tuple>
#include <utility>
#include "asio/async_result.hpp"
#include "asio/is_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

struct use_await_t
{
  template <typename InnerExecutor>
  struct executor_with_default : InnerExecutor
  {
    typedef use_await_t default_completion_token_type;

    executor_with_default(const InnerExecutor& ex) ASIO_NOEXCEPT
      : InnerExecutor(ex)
    {
    }
  };

  template <typename T>
  struct as_default_on_t
  {
    typedef typename T::template rebind_executor<
      executor_with_default<typename T::executor_type> >::other type;
  };

  template <typename T>
  static typename as_default_on_t<typename decay<T>::type>::type
  as_default_on(ASIO_MOVE_ARG(T) object)
  {
    return typename as_default_on_t<typename decay<T>::type>::type(
        ASIO_MOVE_CAST(T)(object));
  }
};

#if !defined(GENERATING_DOCUMENTATION)

template <typename InnerExecutor>
struct is_executor<use_await_t::executor_with_default<InnerExecutor> >
  : is_executor<InnerExecutor>
{
};

#endif // !defined(GENERATING_DOCUMENTATION)

template <typename R, typename... Args>
class async_result<use_await_t, R(Args...)>
{
private:
  template <typename Initiation, typename... InitArgs>
  struct awaitable
  {
    template <typename T>
    class allocator;

    struct handler
    {
      typedef allocator<void> allocator_type;

      allocator_type get_allocator() const
      {
        return allocator_type(awaitable_);
      }

      void operator()(Args... results)
      {
        std::tuple<Args...> result(std::move(results)...);
        awaitable_->result_ = &result;
        awaitable_->owner_.resume();
      }

      awaitable* awaitable_; // The handler has no ownership of the coroutine.
    };

    using storage_type = intermediate_storage_t<
        Initiation, handler, InitArgs...>;

    template <typename T>
    class allocator
    {
    public:
      typedef T value_type;

      explicit allocator(awaitable* a) noexcept
        : awaitable_(a)
      {
      }

      template <typename U>
      allocator(const allocator<U>& a) noexcept
        : awaitable_(a.awaitable_)
      {
      }

      T* allocate(std::size_t n)
      {
        if constexpr (std::is_same_v<storage_type, void>)
        {
          return static_cast<T*>(::operator new(sizeof(T) * n));
        }
        else
        {
          return static_cast<T*>(static_cast<void*>(&awaitable_->storage_));
        }
      }

      void deallocate(T* p, std::size_t)
      {
        if constexpr (std::is_same_v<storage_type, void>)
        {
          ::operator delete(p);
        }
      }

    private:
      template <typename> friend class allocator;
      awaitable* awaitable_;
    };

    bool await_ready() const noexcept
    {
      return false;
    }

    void await_suspend(std::experimental::coroutine_handle<> h) noexcept
    {
      owner_ = h;
      std::apply(
          [&](auto&&... a)
          {
            initiation_(handler{this}, std::forward<decltype(a)>(a)...);
          },
          init_args_
        );
    }

    std::tuple<Args...> await_resume()
    {
      return std::move(*static_cast<std::tuple<Args...>*>(result_));
    }

    Initiation initiation_;
    std::tuple<InitArgs...> init_args_;
    std::experimental::coroutine_handle<> owner_ = nullptr;
    void* result_ = nullptr;
    std::conditional_t<
        std::is_same_v<storage_type, void>,
        char, storage_type> storage_{};
  };

public:
  template <typename Initiation, typename... InitArgs>
  static auto initiate(Initiation initiation,
      use_await_t, InitArgs... init_args)
  {
    return awaitable<Initiation, InitArgs...>{
        std::move(initiation),
        std::forward_as_tuple(std::move(init_args)...)};
  }
};

/// A completion token object that indicates that an Awaitable should be
/// returned.
#if defined(ASIO_HAS_CONSTEXPR) || defined(GENERATING_DOCUMENTATION)
constexpr use_await_t use_await;
#elif defined(ASIO_MSVC)
__declspec(selectany) use_await_t use_await;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_CO_AWAIT) || defined(GENERATING_DOCUMENTATION)

#endif // ASIO_USE_AWAIT_HPP
