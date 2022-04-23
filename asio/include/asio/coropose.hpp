// Copyright (c) 2021 Klemens D. Morgenstern
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ASIO_COROPOSE_HPP
#define ASIO_COROPOSE_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/co_spawn.hpp"
#include "asio/async_result.hpp"
#include "asio/awaitable.hpp"
#include "asio/use_awaitable.hpp"
#include "asio/experimental/coro.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template<typename Signature>
struct async_coropose_tag
{
};

template<typename Signature, typename CompletionToken>
struct coropose_awaitable_frame;

template<typename T, typename CompletionToken>
struct coropose_awaitable_frame<void(T), CompletionToken>: asio::detail::awaitable_frame<T, any_io_executor>
{
  using asio::detail::awaitable_frame<T, any_io_executor>::await_transform;

  template<typename Context,
          typename ... Args>
  coropose_awaitable_frame(Context && ctx, Args && ... args)
  {
    auto l =
            [cpl = std::move(std::get<sizeof...(Args) - 2>(std::forward_as_tuple(args...)))](std::exception_ptr e, T && t) mutable
            {
              if (e)
                std::rethrow_exception(e); // so run() throws e
              std::move(cpl)(std::forward<T>(t));
            };

    asio::co_spawn(std::forward<Context>(ctx),
                   asio::detail::awaitable_frame<T, any_io_executor>::get_return_object(),
                   std::move(l)
                   );
  }

  void get_return_object()
  {
  }
};

template<typename T, typename CompletionToken>
struct coropose_awaitable_frame<void(std::exception_ptr, T), CompletionToken>: asio::detail::awaitable_frame<T, any_io_executor>
{
  using asio::detail::awaitable_frame<T, any_io_executor>::await_transform;

  template<typename Context,
          typename ... Args>
  coropose_awaitable_frame(Context && ctx, Args && ... args)
  {
    asio::co_spawn(std::forward<Context>(ctx),
                   asio::detail::awaitable_frame<T, any_io_executor>::get_return_object(),
                   std::move(std::get<sizeof...(Args) - 2>(std::forward_as_tuple(args...)))
    );
  }

  void get_return_object()
  {
    // NOOP:
  }
};

//template<typename
template<typename Signature, typename CompletionToken>
struct async_coropose_traits
{
  using promise_type = coropose_awaitable_frame<Signature, CompletionToken>;
  using type = void;
};


template<typename T, typename Executor>
struct async_coropose_traits<void(T), asio::use_awaitable_t<Executor>>
{
  using type = asio::awaitable<T, Executor>;
};

template<typename T, typename Executor>
struct async_coropose_traits<void(asio::error_code, T), asio::use_awaitable_t<Executor>>
{
  using type = asio::awaitable<T, Executor>;
};

template<typename T, typename Executor>
struct async_coropose_traits<void(std::exception_ptr, T), asio::use_awaitable_t<Executor>>
{
  using type = asio::awaitable<T, Executor>;
};


template<typename T, typename Executor>
struct async_coropose_traits<void(T), asio::experimental::use_coro_t<Executor>>
{
  using type = asio::experimental::coro<void() noexcept, T, Executor>;
};

template<typename T, typename Executor>
struct async_coropose_traits<void(asio::error_code, T), asio::experimental::use_coro_t<Executor>>
{
  using type = asio::experimental::coro<void() noexcept, std::variant<error_code, T>, Executor>;
};

template<typename T, typename Executor>
struct async_coropose_traits<void(std::exception_ptr, T), asio::experimental::use_coro_t<Executor>>
{
  using type = asio::experimental::coro<void, T, Executor>;
};

template<typename Signature, typename CompletionToken>
using async_coropose_t = typename async_coropose_traits<Signature, std::decay_t<CompletionToken>>::type;

} // namespace asio

#if defined(ASIO_HAS_STD_COROUTINE)
namespace std
#else // defined(ASIO_HAS_STD_COROUTINE)
namespace std::experimental
#endif // defined(ASIO_HAS_STD_COROUTINE)
{

template<typename Context, typename CompletionToken, typename Signature>
struct coroutine_traits<void, Context,
                        CompletionToken, asio::async_coropose_tag<Signature> >
{
  using promise_type = typename asio::async_coropose_traits<Signature, std::decay_t<CompletionToken>>::promise_type;
};


template<typename Context, typename T0, typename CompletionToken, typename Signature>
struct coroutine_traits<void, Context,
        T0,
        CompletionToken, asio::async_coropose_tag<Signature> >
{
  using promise_type = typename asio::async_coropose_traits<Signature, std::decay_t<CompletionToken>>::promise_type;
};

template<typename Context, typename T0, typename T1, typename CompletionToken, typename Signature>
struct coroutine_traits<void, Context,
        T0, T1,
        CompletionToken, asio::async_coropose_tag<Signature> >
{
  using promise_type = typename asio::async_coropose_traits<Signature, std::decay_t<CompletionToken>>::promise_type;
};

template<typename Context, typename T0, typename T1, typename T2, typename CompletionToken, typename Signature>
struct coroutine_traits<void, Context,
        T0, T1, T2,
        CompletionToken, asio::async_coropose_tag<Signature> >
{
  using promise_type = typename asio::async_coropose_traits<Signature, std::decay_t<CompletionToken>>::promise_type;
};

template<typename Context, typename T0, typename T1,  typename T2,  typename T3, typename CompletionToken, typename Signature>
struct coroutine_traits<void, Context,
        T0, T1, T2, T3,
        CompletionToken, asio::async_coropose_tag<Signature> >
{
  using promise_type = typename asio::async_coropose_traits<Signature, std::decay_t<CompletionToken>>::promise_type;
};

template<typename Context,
        typename T0, typename T1, typename T2, typename T3, typename T4,
        typename CompletionToken, typename Signature>
struct coroutine_traits<void, Context,
        T0, T1, T2, T3, T4,
        CompletionToken, asio::async_coropose_tag<Signature> >
{
  using promise_type = typename asio::async_coropose_traits<Signature, std::decay_t<CompletionToken>>::promise_type;
};

template<typename Context,
        typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
        typename CompletionToken, typename Signature>
struct coroutine_traits<void, Context,
        T0, T1, T2, T3, T4, T5,
        CompletionToken, asio::async_coropose_tag<Signature> >
{
  using promise_type = typename asio::async_coropose_traits<Signature, std::decay_t<CompletionToken>>::promise_type;
};

template<typename Context,
        typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
        typename CompletionToken, typename Signature>
struct coroutine_traits<void, Context,
        T0, T1, T2, T3, T4, T5, T6,
        CompletionToken, asio::async_coropose_tag<Signature> >
{
  using promise_type = typename asio::async_coropose_traits<Signature, std::decay_t<CompletionToken>>::promise_type;
};

template<typename Context,
        typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
        typename CompletionToken, typename Signature>
struct coroutine_traits<void, Context,
        T0, T1, T2, T3, T4, T5, T6, T7,
        CompletionToken, asio::async_coropose_tag<Signature> >
{
  using promise_type = typename asio::async_coropose_traits<Signature, std::decay_t<CompletionToken>>::promise_type;
};



}


#include "asio/detail/pop_options.hpp"

#include "asio/impl/compose.hpp"

#endif //ASIO_COROPOSE_HPP
