//
// experimental/deferred.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_DEFERRED_HPP
#define ASIO_EXPERIMENTAL_DEFERRED_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace detail {

struct deferred_empty_branch
{
  template <typename... Args>
  void operator()(ASIO_MOVE_ARG(Args)...) ASIO_RVALUE_REF_QUAL
  {
  }
};

} // namespace detail

template <typename Impl> class deferred_operation;
template <typename OnTrue, typename OnFalse> class deferred_condition;

/// Class used to specify that an asynchronous operation should return a
/// function object to lazily launch the operation.
/**
 * The deferred_t class is used to indicate that an asynchronous operation should
 * return a function object which is itself an initiation function. A deferred_t
 * object may be passed as a completion token to an asynchronous operation,
 * typically using the special value @c asio::deferred. For example:
 *
 * @code auto my_sender
 *   = my_socket.async_read_some(my_buffer,
 *       asio::experimental::deferred); @endcode
 *
 * The initiating function (async_read_some in the above example) returns a
 * function object that will lazily initiate the operation.
 */
class deferred_t
{
public:
  /// Default constructor.
  ASIO_CONSTEXPR deferred_t()
  {
  }

  /// Adapts an executor to add the @c deferred_t completion token as the
  /// default.
  template <typename InnerExecutor>
  struct executor_with_default : InnerExecutor
  {
    /// Specify @c deferred_t as the default completion token type.
    typedef deferred_t default_completion_token_type;

    /// Construct the adapted executor from the inner executor type.
    template <typename InnerExecutor1>
    executor_with_default(const InnerExecutor1& ex,
        typename constraint<
          conditional<
            !is_same<InnerExecutor1, executor_with_default>::value,
            is_convertible<InnerExecutor1, InnerExecutor>,
            false_type
          >::type::value
        >::type = 0) ASIO_NOEXCEPT
      : InnerExecutor(ex)
    {
    }
  };

  /// Type alias to adapt an I/O object to use @c deferred_t as its
  /// default completion token type.
#if defined(ASIO_HAS_ALIAS_TEMPLATES) \
  || defined(GENERATING_DOCUMENTATION)
  template <typename T>
  using as_default_on_t = typename T::template rebind_executor<
      executor_with_default<typename T::executor_type> >::other;
#endif // defined(ASIO_HAS_ALIAS_TEMPLATES)
       //   || defined(GENERATING_DOCUMENTATION)

  /// Function helper to adapt an I/O object to use @c deferred_t as its
  /// default completion token type.
  template <typename T>
  static typename decay<T>::type::template rebind_executor<
      executor_with_default<typename decay<T>::type::executor_type>
    >::other
  as_default_on(ASIO_MOVE_ARG(T) object)
  {
    return typename decay<T>::type::template rebind_executor<
        executor_with_default<typename decay<T>::type::executor_type>
      >::other(ASIO_MOVE_CAST(T)(object));
  }

  /// Creates a conditional object for branching deferred operations.
  static deferred_condition<detail::deferred_empty_branch,
    detail::deferred_empty_branch> when(bool b);

  /// Returns a deferred operation that returns the provided values.
  template <typename... Args>
  static auto values(Args... args);
};

/// Class used to encapsulate the result of a deferred operation.
template <typename Impl>
class deferred_operation : public Impl
{
public:
  template <typename I>
  explicit deferred_operation(ASIO_MOVE_ARG(I) impl)
    : Impl(ASIO_MOVE_CAST(I)(impl))
  {
  }
};

/// Class used to make conditional deferred branches.
template <typename OnTrue, typename OnFalse>
class deferred_condition
{
public:
  template <typename T, typename F>
  explicit deferred_condition(bool b, ASIO_MOVE_ARG(T) on_true,
      ASIO_MOVE_ARG(F) on_false)
    : on_true_(ASIO_MOVE_CAST(T)(on_true)),
      on_false_(ASIO_MOVE_CAST(F)(on_false)),
      bool_(b)
  {
  }

  template <typename... Args>
  auto operator()(ASIO_MOVE_ARG(Args)... args) ASIO_RVALUE_REF_QUAL
  {
    if (bool_)
    {
      return ASIO_MOVE_OR_LVALUE(OnTrue)(on_true_)(
          ASIO_MOVE_CAST(Args)(args)...);
    }
    else
    {
      return ASIO_MOVE_OR_LVALUE(OnFalse)(on_false_)(
          ASIO_MOVE_CAST(Args)(args)...);
    }
  }

  template <typename Impl>
  deferred_operation<deferred_condition<Impl, OnFalse>> then(
      deferred_operation<Impl> op,
      typename constraint<
        is_same<
          typename conditional<true, OnTrue, Impl>::type,
          detail::deferred_empty_branch
        >::value
      >::type* = 0) ASIO_RVALUE_REF_QUAL
  {
    return deferred_operation<deferred_condition<Impl, OnFalse>>(
        deferred_condition<Impl, OnFalse>(bool_,
          ASIO_MOVE_CAST(Impl)(op),
          ASIO_MOVE_CAST(OnFalse)(on_false_)));
  }

  template <typename Impl>
  deferred_operation<deferred_condition<OnTrue, Impl>> otherwise(
      deferred_operation<Impl> op,
      typename constraint<
        !is_same<
          typename conditional<true, OnTrue, Impl>::type,
          detail::deferred_empty_branch
        >::value
      >::type* = 0,
      typename constraint<
        is_same<
          typename conditional<true, OnFalse, Impl>::type,
          detail::deferred_empty_branch
        >::value
      >::type* = 0) ASIO_RVALUE_REF_QUAL
  {
    return deferred_operation<deferred_condition<OnTrue, Impl>>(
        deferred_condition<OnTrue, Impl>(bool_,
          ASIO_MOVE_CAST(OnTrue)(on_true_),
          ASIO_MOVE_CAST(Impl)(op)));
  }

private:
  OnTrue on_true_;
  OnFalse on_false_;
  bool bool_;
};

inline deferred_condition<detail::deferred_empty_branch, detail::deferred_empty_branch>
deferred_t::when(bool b)
{
  return deferred_condition<detail::deferred_empty_branch, detail::deferred_empty_branch>(
      b, detail::deferred_empty_branch(), detail::deferred_empty_branch());
}

template <typename... Args>
auto deferred_t::values(Args... args)
{
  return asio::async_initiate<const deferred_t&, void(Args...)>(
      [](auto handler, auto... args)
      {
        handler(ASIO_MOVE_CAST(Args)(args)...);
      }, deferred_t(), ASIO_MOVE_CAST(Args)(args)...);
}

/// Pipe operator used to chain deferred operations.
template <typename Impl, typename CompletionToken>
auto operator|(deferred_operation<Impl> head,
    ASIO_MOVE_ARG(CompletionToken) tail)
  -> decltype(
    ASIO_MOVE_OR_LVALUE(deferred_operation<Impl>)(head)(
        ASIO_MOVE_CAST(CompletionToken)(tail)))
{
  return ASIO_MOVE_OR_LVALUE(deferred_operation<Impl>)(head)(
      ASIO_MOVE_CAST(CompletionToken)(tail));
}

template <typename T>
struct is_deferred_operation : false_type
{
};

template <typename Impl>
struct is_deferred_operation<deferred_operation<Impl>> : true_type
{
};

/// A special value, similar to std::nothrow.
/**
 * See the documentation for asio::experimental::deferred_t for a usage example.
 */
#if defined(ASIO_HAS_CONSTEXPR) || defined(GENERATING_DOCUMENTATION)
constexpr deferred_t deferred;
#elif defined(ASIO_MSVC)
__declspec(selectany) deferred_t deferred;
#endif

} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/experimental/impl/deferred.hpp"

#endif // ASIO_EXPERIMENTAL_DEFERRED_HPP
