//
// compose.hpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_COMPOSE_HPP
#define ASIO_COMPOSE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/associated_executor.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/base_from_cancellation_state.hpp"
#include "asio/detail/composed_work.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_cont_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)
template <typename Impl, typename Work, typename Handler, typename Signature>
class composed_op;

template <typename Impl, typename Work, typename Handler,
    typename R, typename... Args>
class composed_op<Impl, Work, Handler, R(Args...)>
#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)
template <typename Impl, typename Work, typename Handler, typename Signature>
class composed_op
#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)
  : public base_from_cancellation_state<Handler>
{
public:
  template <typename I, typename W, typename H>
  composed_op(ASIO_MOVE_ARG(I) impl,
      ASIO_MOVE_ARG(W) work,
      ASIO_MOVE_ARG(H) handler)
    : base_from_cancellation_state<Handler>(
        handler, enable_terminal_cancellation()),
      impl_(ASIO_MOVE_CAST(I)(impl)),
      work_(ASIO_MOVE_CAST(W)(work)),
      handler_(ASIO_MOVE_CAST(H)(handler)),
      invocations_(0)
  {
  }

#if defined(ASIO_HAS_MOVE)
  composed_op(composed_op&& other)
    : base_from_cancellation_state<Handler>(
        ASIO_MOVE_CAST(base_from_cancellation_state<
          Handler>)(other)),
      impl_(ASIO_MOVE_CAST(Impl)(other.impl_)),
      work_(ASIO_MOVE_CAST(Work)(other.work_)),
      handler_(ASIO_MOVE_CAST(Handler)(other.handler_)),
      invocations_(other.invocations_)
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  typedef typename composed_work_guard<
    typename Work::head_type>::executor_type io_executor_type;

  io_executor_type get_io_executor() const ASIO_NOEXCEPT
  {
    return work_.head_.get_executor();
  }

  typedef typename associated_executor<Handler, io_executor_type>::type
    executor_type;

  executor_type get_executor() const ASIO_NOEXCEPT
  {
    return (get_associated_executor)(handler_, work_.head_.get_executor());
  }

  typedef typename associated_allocator<Handler,
    std::allocator<void> >::type allocator_type;

  allocator_type get_allocator() const ASIO_NOEXCEPT
  {
    return (get_associated_allocator)(handler_, std::allocator<void>());
  }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template<typename... T>
  void operator()(ASIO_MOVE_ARG(T)... t)
  {
    if (invocations_ < ~0u)
      ++invocations_;
    this->get_cancellation_state().slot().clear();
    impl_(*this, ASIO_MOVE_CAST(T)(t)...);
  }

  void complete(Args... args)
  {
    this->work_.reset();
    ASIO_MOVE_OR_LVALUE(Handler)(this->handler_)(
        ASIO_MOVE_CAST(Args)(args)...);
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  void operator()()
  {
    if (invocations_ < ~0u)
      ++invocations_;
    this->get_cancellation_state().slot().clear();
    impl_(*this);
  }

  void complete()
  {
    this->work_.reset();
    ASIO_MOVE_OR_LVALUE(Handler)(this->handler_)();
  }

#define ASIO_PRIVATE_COMPOSED_OP_DEF(n) \
  template<ASIO_VARIADIC_TPARAMS(n)> \
  void operator()(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    if (invocations_ < ~0u) \
      ++invocations_; \
    this->get_cancellation_state().slot().clear(); \
    impl_(*this, ASIO_VARIADIC_MOVE_ARGS(n)); \
  } \
  \
  template<ASIO_VARIADIC_TPARAMS(n)> \
  void complete(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    this->work_.reset(); \
    ASIO_MOVE_OR_LVALUE(Handler)(this->handler_)( \
        ASIO_VARIADIC_MOVE_ARGS(n)); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_COMPOSED_OP_DEF)
#undef ASIO_PRIVATE_COMPOSED_OP_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  void reset_cancellation_state()
  {
    base_from_cancellation_state<Handler>::reset_cancellation_state(handler_);
  }

  template <typename Filter>
  void reset_cancellation_state(ASIO_MOVE_ARG(Filter) filter)
  {
    base_from_cancellation_state<Handler>::reset_cancellation_state(handler_,
        ASIO_MOVE_CAST(Filter)(filter));
  }

  template <typename InFilter, typename OutFilter>
  void reset_cancellation_state(ASIO_MOVE_ARG(InFilter) in_filter,
      ASIO_MOVE_ARG(OutFilter) out_filter)
  {
    base_from_cancellation_state<Handler>::reset_cancellation_state(handler_,
        ASIO_MOVE_CAST(InFilter)(in_filter),
        ASIO_MOVE_CAST(OutFilter)(out_filter));
  }

  cancellation_type_t cancelled() const ASIO_NOEXCEPT
  {
    return base_from_cancellation_state<Handler>::cancelled();
  }

//private:
  Impl impl_;
  Work work_;
  Handler handler_;
  unsigned invocations_;
};

template <typename Impl, typename Work, typename Handler, typename Signature>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
    composed_op<Impl, Work, Handler, Signature>* this_handler)
{
#if defined(ASIO_NO_DEPRECATED)
  asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
  return asio_handler_allocate_is_no_longer_used();
#else // defined(ASIO_NO_DEPRECATED)
  return asio_handler_alloc_helpers::allocate(
      size, this_handler->handler_);
#endif // defined(ASIO_NO_DEPRECATED)
}

template <typename Impl, typename Work, typename Handler, typename Signature>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
    composed_op<Impl, Work, Handler, Signature>* this_handler)
{
  asio_handler_alloc_helpers::deallocate(
      pointer, size, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
  return asio_handler_deallocate_is_no_longer_used();
#endif // defined(ASIO_NO_DEPRECATED)
}

template <typename Impl, typename Work, typename Handler, typename Signature>
inline bool asio_handler_is_continuation(
    composed_op<Impl, Work, Handler, Signature>* this_handler)
{
  return this_handler->invocations_ > 1 ? true
    : asio_handler_cont_helpers::is_continuation(
        this_handler->handler_);
}

template <typename Function, typename Impl,
    typename Work, typename Handler, typename Signature>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
    composed_op<Impl, Work, Handler, Signature>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
  return asio_handler_invoke_is_no_longer_used();
#endif // defined(ASIO_NO_DEPRECATED)
}

template <typename Function, typename Impl,
    typename Work, typename Handler, typename Signature>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
    composed_op<Impl, Work, Handler, Signature>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
  return asio_handler_invoke_is_no_longer_used();
#endif // defined(ASIO_NO_DEPRECATED)
}

template <typename Signature, typename Executors>
class initiate_composed_op
{
public:
  typedef typename composed_io_executors<Executors>::head_type executor_type;

  template <typename T>
  explicit initiate_composed_op(int, ASIO_MOVE_ARG(T) executors)
    : executors_(ASIO_MOVE_CAST(T)(executors))
  {
  }

  executor_type get_executor() const ASIO_NOEXCEPT
  {
    return executors_.head_;
  }

  template <typename Handler, typename Impl>
  void operator()(ASIO_MOVE_ARG(Handler) handler,
      ASIO_MOVE_ARG(Impl) impl) const
  {
    composed_op<typename decay<Impl>::type, composed_work<Executors>,
      typename decay<Handler>::type, Signature>(
        ASIO_MOVE_CAST(Impl)(impl),
        composed_work<Executors>(executors_),
        ASIO_MOVE_CAST(Handler)(handler))();
  }

private:
  composed_io_executors<Executors> executors_;
};

template <typename Signature, typename Executors>
inline initiate_composed_op<Signature, Executors> make_initiate_composed_op(
    ASIO_MOVE_ARG(composed_io_executors<Executors>) executors)
{
  return initiate_composed_op<Signature, Executors>(0,
      ASIO_MOVE_CAST(composed_io_executors<Executors>)(executors));
}

} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <template <typename, typename> class Associator,
    typename Impl, typename Work, typename Handler,
    typename Signature, typename DefaultCandidate>
struct associator<Associator,
    detail::composed_op<Impl, Work, Handler, Signature>,
    DefaultCandidate>
  : Associator<Handler, DefaultCandidate>
{
  static typename Associator<Handler, DefaultCandidate>::type
  get(const detail::composed_op<Impl, Work, Handler, Signature>& h)
    ASIO_NOEXCEPT
  {
    return Associator<Handler, DefaultCandidate>::get(h.handler_);
  }

  static ASIO_AUTO_RETURN_TYPE_PREFIX2(
      typename Associator<Handler, DefaultCandidate>::type)
  get(const detail::composed_op<Impl, Work, Handler, Signature>& h,
      const DefaultCandidate& c) ASIO_NOEXCEPT
    ASIO_AUTO_RETURN_TYPE_SUFFIX((
      Associator<Handler, DefaultCandidate>::get(h.handler_, c)))
  {
    return Associator<Handler, DefaultCandidate>::get(h.handler_, c);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

#if defined(ASIO_HAS_VARIADIC_TEMPLATES) \
  || defined(GENERATING_DOCUMENTATION)

/// Launch an asynchronous operation with a stateful implementation.
/**
 * The async_compose function simplifies the implementation of composed
 * asynchronous operations automatically wrapping a stateful function object
 * with a conforming intermediate completion handler.
 *
 * @param implementation A function object that contains the implementation of
 * the composed asynchronous operation. The first argument to the function
 * object is a non-const reference to the enclosing intermediate completion
 * handler. The remaining arguments are any arguments that originate from the
 * completion handlers of any asynchronous operations performed by the
 * implementation.
 *
 * @param token The completion token.
 *
 * @param io_objects_or_executors Zero or more I/O objects or I/O executors for
 * which outstanding work must be maintained.
 *
 * @par Per-Operation Cancellation
 * By default, terminal per-operation cancellation is enabled for
 * composed operations that are implemented using @c async_compose. To
 * disable cancellation for the composed operation, or to alter its
 * supported cancellation types, call the @c self object's @c
 * reset_cancellation_state function.
 *
 * @par Example:
 *
 * @code struct async_echo_implementation
 * {
 *   tcp::socket& socket_;
 *   asio::mutable_buffer buffer_;
 *   enum { starting, reading, writing } state_;
 *
 *   template <typename Self>
 *   void operator()(Self& self,
 *       asio::error_code error = {},
 *       std::size_t n = 0)
 *   {
 *     switch (state_)
 *     {
 *     case starting:
 *       state_ = reading;
 *       socket_.async_read_some(
 *           buffer_, std::move(self));
 *       break;
 *     case reading:
 *       if (error)
 *       {
 *         self.complete(error, 0);
 *       }
 *       else
 *       {
 *         state_ = writing;
 *         asio::async_write(socket_, buffer_,
 *             asio::transfer_exactly(n),
 *             std::move(self));
 *       }
 *       break;
 *     case writing:
 *       self.complete(error, n);
 *       break;
 *     }
 *   }
 * };
 *
 * template <typename CompletionToken>
 * auto async_echo(tcp::socket& socket,
 *     asio::mutable_buffer buffer,
 *     CompletionToken&& token) ->
 *   decltype(
 *     asio::async_compose<CompletionToken,
 *       void(asio::error_code, std::size_t)>(
 *         std::declval<async_echo_implementation>(),
 *         token, socket))
 * {
 *   return asio::async_compose<CompletionToken,
 *     void(asio::error_code, std::size_t)>(
 *       async_echo_implementation{socket, buffer,
 *         async_echo_implementation::starting},
 *       token, socket);
 * } @endcode
 */
template <typename CompletionToken, typename Signature,
    typename Implementation, typename... IoObjectsOrExecutors>
ASIO_INITFN_AUTO_RESULT_TYPE_PREFIX(CompletionToken, Signature)
async_compose(ASIO_MOVE_ARG(Implementation) implementation,
    ASIO_NONDEDUCED_MOVE_ARG(CompletionToken) token,
    ASIO_MOVE_ARG(IoObjectsOrExecutors)... io_objects_or_executors)
  ASIO_INITFN_AUTO_RESULT_TYPE_SUFFIX((
    async_initiate<CompletionToken, Signature>(
        detail::make_initiate_composed_op<Signature>(
          detail::make_composed_io_executors(
            detail::get_composed_io_executor(
              ASIO_MOVE_CAST(IoObjectsOrExecutors)(
                io_objects_or_executors))...)),
        token, ASIO_MOVE_CAST(Implementation)(implementation))))
{
  return async_initiate<CompletionToken, Signature>(
      detail::make_initiate_composed_op<Signature>(
        detail::make_composed_io_executors(
          detail::get_composed_io_executor(
            ASIO_MOVE_CAST(IoObjectsOrExecutors)(
              io_objects_or_executors))...)),
      token, ASIO_MOVE_CAST(Implementation)(implementation));
}

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)
      //   || defined(GENERATING_DOCUMENTATION)

template <typename CompletionToken, typename Signature, typename Implementation>
ASIO_INITFN_AUTO_RESULT_TYPE_PREFIX(CompletionToken, Signature)
async_compose(ASIO_MOVE_ARG(Implementation) implementation,
    ASIO_NONDEDUCED_MOVE_ARG(CompletionToken) token)
  ASIO_INITFN_AUTO_RESULT_TYPE_SUFFIX((
    async_initiate<CompletionToken, Signature>(
        detail::make_initiate_composed_op<Signature>(
          detail::make_composed_io_executors()),
        token, ASIO_MOVE_CAST(Implementation)(implementation))))
{
  return async_initiate<CompletionToken, Signature>(
      detail::make_initiate_composed_op<Signature>(
        detail::make_composed_io_executors()),
      token, ASIO_MOVE_CAST(Implementation)(implementation));
}

# define ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR(n) \
  ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_##n

# define ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_1 \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T1)(x1))
# define ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_2 \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T1)(x1)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T2)(x2))
# define ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_3 \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T1)(x1)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T2)(x2)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T3)(x3))
# define ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_4 \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T1)(x1)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T2)(x2)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T3)(x3)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T4)(x4))
# define ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_5 \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T1)(x1)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T2)(x2)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T3)(x3)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T4)(x4)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T5)(x5))
# define ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_6 \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T1)(x1)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T2)(x2)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T3)(x3)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T4)(x4)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T5)(x5)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T6)(x6))
# define ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_7 \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T1)(x1)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T2)(x2)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T3)(x3)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T4)(x4)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T5)(x5)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T6)(x6)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T7)(x7))
# define ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_8 \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T1)(x1)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T2)(x2)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T3)(x3)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T4)(x4)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T5)(x5)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T6)(x6)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T7)(x7)), \
  detail::get_composed_io_executor(ASIO_MOVE_CAST(T8)(x8))

#define ASIO_PRIVATE_ASYNC_COMPOSE_DEF(n) \
  template <typename CompletionToken, typename Signature, \
      typename Implementation, ASIO_VARIADIC_TPARAMS(n)> \
  ASIO_INITFN_AUTO_RESULT_TYPE_PREFIX(CompletionToken, Signature) \
  async_compose(ASIO_MOVE_ARG(Implementation) implementation, \
      ASIO_NONDEDUCED_MOVE_ARG(CompletionToken) token, \
      ASIO_VARIADIC_MOVE_PARAMS(n)) \
    ASIO_INITFN_AUTO_RESULT_TYPE_SUFFIX(( \
      async_initiate<CompletionToken, Signature>( \
          detail::make_initiate_composed_op<Signature>( \
            detail::make_composed_io_executors( \
              ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR(n))), \
          token, ASIO_MOVE_CAST(Implementation)(implementation)))) \
  { \
    return async_initiate<CompletionToken, Signature>( \
        detail::make_initiate_composed_op<Signature>( \
          detail::make_composed_io_executors( \
            ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR(n))), \
        token, ASIO_MOVE_CAST(Implementation)(implementation)); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_ASYNC_COMPOSE_DEF)
#undef ASIO_PRIVATE_ASYNC_COMPOSE_DEF

#undef ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR
#undef ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_1
#undef ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_2
#undef ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_3
#undef ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_4
#undef ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_5
#undef ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_6
#undef ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_7
#undef ASIO_PRIVATE_GET_COMPOSED_IO_EXECUTOR_8

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)
       //   || defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_COMPOSE_HPP
