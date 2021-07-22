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
#include <tuple>
#include "asio/associator.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {

/// Trait for detecting objects that are usable as deferred operations.
template <typename T>
struct is_deferred : false_type
{
};

/// Type for passing around a signature pack.
template <typename... Signatures>
struct deferred_signatures
{
};

namespace detail {

template <typename SignaturePack1, typename SignaturePack2>
struct cat_deferred_signatures;

template <typename... Sigs1, typename... Sigs2>
struct cat_deferred_signatures<
    deferred_signatures<Sigs1...>,
    deferred_signatures<Sigs2...> >
{
  typedef deferred_signatures<Sigs1..., Sigs2...> type;
};

template <typename... SignaturePacks>
struct merge_deferred_signatures;

template <typename Head1, typename... Tail1>
struct merge_deferred_signatures<
    deferred_signatures<Head1, Tail1...> >
{
  typedef deferred_signatures<Head1, Tail1...> type;
};

template <>
struct merge_deferred_signatures<
    deferred_signatures<>,
    deferred_signatures<> >
{
  typedef deferred_signatures<> type;
};

template <typename Head1, typename... Tail1>
struct merge_deferred_signatures<
    deferred_signatures<Head1, Tail1...>,
    deferred_signatures<> >
{
  typedef deferred_signatures<Head1, Tail1...> type;
};

template <typename Head2, typename... Tail2>
struct merge_deferred_signatures<
    deferred_signatures<>,
    deferred_signatures<Head2, Tail2...> >
{
  typedef deferred_signatures<Head2, Tail2...> type;
};

template <typename Head1, typename... Tail1>
struct merge_deferred_signatures<
    deferred_signatures<Head1, Tail1...>,
    deferred_signatures<Head1> >
{
  typedef deferred_signatures<Head1, Tail1...> type;
};

template <typename Head1, typename... Tail1, typename Head2>
struct merge_deferred_signatures<
    deferred_signatures<Head1, Tail1...>,
    deferred_signatures<Head2> > :
  cat_deferred_signatures<
    deferred_signatures<Head1>,
    typename merge_deferred_signatures<
      deferred_signatures<Tail1...>,
      deferred_signatures<Head2>
    >::type
  >
{
};

template <typename Head1, typename...Tail1, typename Head2, typename... Tail2>
struct merge_deferred_signatures<
    deferred_signatures<Head1, Tail1...>,
    deferred_signatures<Head2, Tail2...> > :
  merge_deferred_signatures<
    typename merge_deferred_signatures<
      deferred_signatures<Head1, Tail1...>,
      deferred_signatures<Head2>
    >::type,
    deferred_signatures<Tail2...>
  >
{
};

template <typename Sigs1, typename Sigs2, typename... SigsN>
struct merge_deferred_signatures<Sigs1, Sigs2, SigsN...> :
  merge_deferred_signatures<
    typename merge_deferred_signatures<Sigs1, Sigs2>::type,
    SigsN...
  >
{
};

} // namespace detail

#if defined(ASIO_HAS_CONCEPTS)

namespace detail {

template <typename T, typename SignaturePack>
struct is_completion_token_for_deferred_signatures : false_type
{
};

template <typename T, typename... Signatures>
struct is_completion_token_for_deferred_signatures<
    T, deferred_signatures<Signatures...> >
  : integral_constant<bool, completion_token_for<T, Signatures...> >
{
};

} // namespace detail

template <typename T, typename SignaturePack>
ASIO_CONCEPT completion_token_for_deferred_signatures =
  detail::is_completion_token_for_deferred_signatures<T, SignaturePack>::value;

#define ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS(sig) \
  ::asio::experimental::completion_token_for_deferred_signatures<sig>
#define ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS2(sig0, sig1) \
  ::asio::experimental::completion_token_for_deferred_signatures<sig0, sig1>
#define ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS3(sig0, sig1, sig2) \
  ::asio::experimental::completion_token_for_deferred_signatures<sig0, sig1, sig2>

#else // defined(ASIO_HAS_CONCEPTS)

#define ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS(sig) \
  typename
#define ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS2(sig0, sig1) \
  typename
#define ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS3(sig0, sig1, sig2) \
  typename

#endif // defined(ASIO_HAS_CONCEPTS)

namespace detail {

// Helper trait for getting the completion signature from an async operation.

struct deferred_signatures_probe {};

template <typename... Signatures>
struct deferred_signatures_probe_result
{
  typedef deferred_signatures<Signatures...> type;
};

template <typename T>
struct deferred_signatures_for
{
  typedef typename decltype(
      declval<T>()(declval<deferred_signatures_probe>()))::type type;
};

// Helper trait for getting the completion signature of the tail in a sequence
// when invoked with the specified arguments.

template <typename HeadSignature, typename Tail>
struct deferred_sequence_signatures_1;

template <typename R, typename... Args, typename Tail>
struct deferred_sequence_signatures_1<R(Args...), Tail>
{
  static_assert(
      !is_same<decltype(declval<Tail>()(declval<Args>()...)), void>::value,
      "deferred functions must produce a deferred return type");

  typedef typename decltype(
      declval<Tail>()(declval<Args>()...)(
        declval<deferred_signatures_probe>()))::type type;
};

template <typename SignaturePack, typename Tail>
struct deferred_sequence_signatures;

template <typename... Signatures, typename Tail>
struct deferred_sequence_signatures<deferred_signatures<Signatures...>, Tail>
{
  typedef typename detail::merge_deferred_signatures<
      typename deferred_sequence_signatures_1<Signatures, Tail>::type...
    >::type type;
};

// Completion handler for the head component of a deferred sequence.
template <typename Handler, typename Tail>
class deferred_sequence_handler
{
public:
  template <typename H, typename T>
  explicit deferred_sequence_handler(
      ASIO_MOVE_ARG(H) handler, ASIO_MOVE_ARG(T) tail)
    : handler_(ASIO_MOVE_CAST(H)(handler)),
      tail_(ASIO_MOVE_CAST(T)(tail))
  {
  }

  template <typename... Args>
  void operator()(ASIO_MOVE_ARG(Args)... args)
  {
    ASIO_MOVE_OR_LVALUE(Tail)(tail_)(
        ASIO_MOVE_CAST(Args)(args)...)(
          ASIO_MOVE_OR_LVALUE(Handler)(handler_));
  }

//private:
  Handler handler_;
  Tail tail_;
};

} // namespace detail

/// Used to represent an empty deferred action.
struct deferred_noop
{
  /// No effect.
  template <typename... Args>
  void operator()(ASIO_MOVE_ARG(Args)...) ASIO_RVALUE_REF_QUAL
  {
  }

#if defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)
  /// No effect.
  template <typename... Args>
  decltype(auto) operator()(ASIO_MOVE_ARG(Args)...) const &
  {
  }
#endif // defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)
};

#if !defined(GENERATING_DOCUMENTATION)
template <>
struct is_deferred<deferred_noop> : true_type
{
};
#endif // !defined(GENERATING_DOCUMENTATION)

namespace detail {

template <>
struct deferred_signatures_for<deferred_noop>
{
  typedef deferred_signatures<> type;
};

} // namespace detail

/// Tag type to disambiguate deferred constructors.
struct deferred_init_tag {};

/// Wraps a function object so that it may be used as an element in a deferred
/// composition.
template <typename Function>
class deferred_function
{
public:
  /// Constructor. 
  template <typename F>
  ASIO_CONSTEXPR explicit deferred_function(
      deferred_init_tag, ASIO_MOVE_ARG(F) function)
    : function_(ASIO_MOVE_CAST(F)(function))
  {
  }

  template <typename... Args>
  decltype(auto) operator()(
      ASIO_MOVE_ARG(Args)... args) ASIO_RVALUE_REF_QUAL
  {
    return ASIO_MOVE_CAST(Function)(function_)(
        ASIO_MOVE_CAST(Args)(args)...);
  }

#if defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)
  template <typename... Args>
  decltype(auto) operator()(
      ASIO_MOVE_ARG(Args)... args) const &
  {
    return deferred_function(*this)(
        ASIO_MOVE_CAST(Args)(args)...);
  }
#endif // defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)

//private:
  Function function_;
};

#if !defined(GENERATING_DOCUMENTATION)
template <typename Function>
struct is_deferred<deferred_function<Function> > : true_type
{
};
#endif // !defined(GENERATING_DOCUMENTATION)

/// Encapsulates deferred values.
template <typename... Values>
class ASIO_NODISCARD deferred_values
{
private:
  std::tuple<Values...> values_;

  struct initiate
  {
    template <typename Handler, typename... V>
    void operator()(Handler handler, ASIO_MOVE_ARG(V)... values)
    {
      ASIO_MOVE_OR_LVALUE(Handler)(handler)(
          ASIO_MOVE_CAST(V)(values)...);
    }
  };

  template <typename CompletionToken, std::size_t... I>
  decltype(auto) invoke_helper(
      ASIO_MOVE_ARG(CompletionToken) token,
      std::index_sequence<I...>)
  {
    return asio::async_initiate<CompletionToken, void(Values...)>(
        initiate(), token,
        std::get<I>(ASIO_MOVE_CAST(std::tuple<Values...>)(values_))...);
  }

public:
  /// Construct a deferred asynchronous operation from the arguments to an
  /// initiation function object.
  template <typename... V>
  ASIO_CONSTEXPR explicit deferred_values(
      deferred_init_tag, ASIO_MOVE_ARG(V)... values)
    : values_(ASIO_MOVE_CAST(V)(values)...)
  {
  }

  /// Initiate the deferred operation using the supplied completion token.
  template <ASIO_COMPLETION_TOKEN_FOR(void(Values...)) CompletionToken>
  decltype(auto) operator()(
      ASIO_MOVE_ARG(CompletionToken) token) ASIO_RVALUE_REF_QUAL
  {
    return this->invoke_helper(
        ASIO_MOVE_CAST(CompletionToken)(token),
        std::index_sequence_for<Values...>());
  }

#if defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)
  template <ASIO_COMPLETION_TOKEN_FOR(void(Values...)) CompletionToken>
  decltype(auto) operator()(
      ASIO_MOVE_ARG(CompletionToken) token) const &
  {
    return deferred_values(*this)(
        ASIO_MOVE_CAST(CompletionToken)(token));
  }
#endif // defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)
};

#if !defined(GENERATING_DOCUMENTATION)
template <typename... Values>
struct is_deferred<deferred_values<Values...> > : true_type
{
};
#endif // !defined(GENERATING_DOCUMENTATION)

/// Encapsulates a deferred asynchronous operation.
template <typename SignaturePack, typename Initiation, typename... InitArgs>
class ASIO_NODISCARD deferred_async_operation
{
private:
  typename decay<Initiation>::type initiation_;
  typedef std::tuple<typename decay<InitArgs>::type...> init_args_t;
  init_args_t init_args_;

  template <typename... S,
      ASIO_COMPLETION_TOKEN_FOR(S...) CompletionToken,
      std::size_t... I>
  decltype(auto) invoke_helper(deferred_signatures<S...>,
      ASIO_MOVE_ARG(CompletionToken) token, std::index_sequence<I...>)
  {
    return asio::async_initiate<CompletionToken, S...>(
        ASIO_MOVE_CAST(typename decay<Initiation>::type)(initiation_),
        token, std::get<I>(ASIO_MOVE_CAST(init_args_t)(init_args_))...);
  }

public:
  /// Construct a deferred asynchronous operation from the arguments to an
  /// initiation function object.
  template <typename I, typename... A>
  ASIO_CONSTEXPR explicit deferred_async_operation(
      deferred_init_tag, ASIO_MOVE_ARG(I) initiation,
      ASIO_MOVE_ARG(A)... init_args)
    : initiation_(ASIO_MOVE_CAST(I)(initiation)),
      init_args_(ASIO_MOVE_CAST(A)(init_args)...)
  {
  }

  /// Initiate the asynchronous operation using the supplied completion token.
  template <
      ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS(SignaturePack)
        CompletionToken>
  decltype(auto) operator()(
      ASIO_MOVE_ARG(CompletionToken) token) ASIO_RVALUE_REF_QUAL
  {
    return this->invoke_helper(SignaturePack{},
        ASIO_MOVE_CAST(CompletionToken)(token),
        std::index_sequence_for<InitArgs...>());
  }

#if defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)
  template <
      ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS(SignaturePack)
        CompletionToken>
  decltype(auto) operator()(
      ASIO_MOVE_ARG(CompletionToken) token) const &
  {
    return deferred_async_operation(*this)(
        ASIO_MOVE_CAST(CompletionToken)(token));
  }
#endif // defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)
};

#if !defined(GENERATING_DOCUMENTATION)
template <typename SignaturePack, typename Initiation, typename... InitArgs>
struct is_deferred<
    deferred_async_operation<SignaturePack, Initiation, InitArgs...> > :
  true_type
{
};
#endif // !defined(GENERATING_DOCUMENTATION)

/// Defines a link between two consecutive operations in a sequence.
template <typename Head, typename Tail>
class ASIO_NODISCARD deferred_sequence
{
private:
  typedef typename detail::deferred_sequence_signatures<
    typename detail::deferred_signatures_for<Head>::type, Tail>::type
      signatures;

  template <typename... S,
      ASIO_COMPLETION_TOKEN_FOR(S...) CompletionToken>
  decltype(auto) invoke_helper(deferred_signatures<S...>,
      ASIO_MOVE_ARG(CompletionToken) token)
  {
    return asio::async_initiate<CompletionToken, S...>(
        initiate(), token, ASIO_MOVE_OR_LVALUE(Head)(head_),
        ASIO_MOVE_OR_LVALUE(Tail)(tail_));
  }

public:
  template <typename H, typename T>
  ASIO_CONSTEXPR explicit deferred_sequence(deferred_init_tag,
      ASIO_MOVE_ARG(H) head, ASIO_MOVE_ARG(T) tail)
    : head_(ASIO_MOVE_CAST(H)(head)),
      tail_(ASIO_MOVE_CAST(T)(tail))
  {
  }

  template <
      ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS(signatures)
        CompletionToken>
  decltype(auto) operator()(
      ASIO_MOVE_ARG(CompletionToken) token) ASIO_RVALUE_REF_QUAL
  {
    return this->invoke_helper(signatures{},
        ASIO_MOVE_CAST(CompletionToken)(token));
  }

#if defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)
  template <
      ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS(signatures)
        CompletionToken>
  decltype(auto) operator()(
      ASIO_MOVE_ARG(CompletionToken) token) const &
  {
    return deferred_sequence(*this)(
        ASIO_MOVE_CAST(CompletionToken)(token));
  }
#endif // defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)

private:
  struct initiate
  {
    template <typename Handler>
    void operator()(ASIO_MOVE_ARG(Handler) handler,
        Head head, ASIO_MOVE_ARG(Tail) tail)
    {
      ASIO_MOVE_OR_LVALUE(Head)(head)(
          detail::deferred_sequence_handler<
            typename decay<Handler>::type,
            typename decay<Tail>::type>(
              ASIO_MOVE_CAST(Handler)(handler),
              ASIO_MOVE_CAST(Tail)(tail)));
    }
  };

  Head head_;
  Tail tail_;
};

#if !defined(GENERATING_DOCUMENTATION)
template <typename Head, typename Tail>
struct is_deferred<deferred_sequence<Head, Tail> > : true_type
{
};
#endif // !defined(GENERATING_DOCUMENTATION)

/// Used to represent a deferred conditional branch.
template <typename OnTrue = deferred_noop,
    typename OnFalse = deferred_noop>
class ASIO_NODISCARD deferred_conditional
{
private:
  template <typename... S,
      ASIO_COMPLETION_TOKEN_FOR(S...) CompletionToken>
  decltype(auto) invoke_helper(deferred_signatures<S...>,
      ASIO_MOVE_ARG(CompletionToken) token)
  {
    return asio::async_initiate<CompletionToken, S...>(
        initiate(), token, ASIO_MOVE_OR_LVALUE(OnTrue)(on_true_),
        ASIO_MOVE_OR_LVALUE(OnFalse)(on_false_), bool_);
  }

public:
  /// Construct a deferred conditional with the value to determine which branch
  /// will be executed.
  ASIO_CONSTEXPR explicit deferred_conditional(bool b)
    : on_true_(),
      on_false_(),
      bool_(b)
  {
  }

  /// Invoke the conditional branch bsaed on the stored alue.
  template <
    ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS2(
      typename detail::merge_deferred_signatures<
        typename detail::deferred_signatures_for<OnTrue>::type,
        typename detail::deferred_signatures_for<OnFalse>::type
      >::type) CompletionToken>
  decltype(auto) operator()(
      ASIO_MOVE_ARG(CompletionToken) token) ASIO_RVALUE_REF_QUAL
  {
    return this->invoke_helper(
        typename detail::merge_deferred_signatures<
          typename detail::deferred_signatures_for<OnTrue>::type,
          typename detail::deferred_signatures_for<OnFalse>::type
        >::type(), ASIO_MOVE_CAST(CompletionToken)(token));
  }

#if defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)
  template <
    ASIO_COMPLETION_TOKEN_FOR_DEFERRED_SIGS2(
      typename detail::merge_deferred_signatures<
        typename detail::deferred_signatures_for<OnTrue>::type,
        typename detail::deferred_signatures_for<OnFalse>::type
      >::type) CompletionToken>
  decltype(auto) operator()(
      ASIO_MOVE_ARG(CompletionToken) token) const &
  {
    return deferred_conditional(*this)(
        ASIO_MOVE_CAST(CompletionToken)(token));
  }
#endif // defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)

  /// Set the true branch of the conditional.
  template <typename T>
  deferred_conditional<T, OnFalse> then(T on_true,
      typename constraint<
        is_deferred<T>::value
      >::type* = 0,
      typename constraint<
        is_same<
          typename conditional<true, OnTrue, T>::type,
          deferred_noop
        >::value
      >::type* = 0) ASIO_RVALUE_REF_QUAL
  {
    return deferred_conditional<T, OnFalse>(
        bool_, ASIO_MOVE_CAST(T)(on_true),
        ASIO_MOVE_CAST(OnFalse)(on_false_));
  }

  /// Set the false branch of the conditional.
  template <typename T>
  deferred_conditional<OnTrue, T> otherwise(T on_false,
      typename constraint<
        is_deferred<T>::value
      >::type* = 0,
      typename constraint<
        !is_same<
          typename conditional<true, OnTrue, T>::type,
          deferred_noop
        >::value
      >::type* = 0,
      typename constraint<
        is_same<
          typename conditional<true, OnFalse, T>::type,
          deferred_noop
        >::value
      >::type* = 0) ASIO_RVALUE_REF_QUAL
  {
    return deferred_conditional<OnTrue, T>(
        bool_, ASIO_MOVE_CAST(OnTrue)(on_true_),
        ASIO_MOVE_CAST(T)(on_false));
  }

private:
  template <typename T, typename F> friend class deferred_conditional;

  // Helper constructor.
  template <typename T, typename F>
  explicit deferred_conditional(bool b, ASIO_MOVE_ARG(T) on_true,
      ASIO_MOVE_ARG(F) on_false)
    : on_true_(ASIO_MOVE_CAST(T)(on_true)),
      on_false_(ASIO_MOVE_CAST(F)(on_false)),
      bool_(b)
  {
  }

  struct initiate
  {
    template <typename Handler>
    void operator()(ASIO_MOVE_ARG(Handler) handler,
       OnTrue on_true, OnFalse on_false, bool b)
    {
      if (b)
        ASIO_MOVE_OR_LVALUE(OnTrue)(on_true)(
            ASIO_MOVE_CAST(Handler)(handler));
      else
        ASIO_MOVE_OR_LVALUE(OnFalse)(on_false)(
            ASIO_MOVE_CAST(Handler)(handler));
    }
  };

  OnTrue on_true_;
  OnFalse on_false_;
  bool bool_;
};

#if !defined(GENERATING_DOCUMENTATION)
template <typename OnTrue, typename OnFalse>
struct is_deferred<deferred_conditional<OnTrue, OnFalse> > : true_type
{
};
#endif // !defined(GENERATING_DOCUMENTATION)

/// Class used to specify that an asynchronous operation should return a
/// function object to lazily launch the operation.
/**
 * The deferred_t class is used to indicate that an asynchronous operation
 * should return a function object which is itself an initiation function. A
 * deferred_t object may be passed as a completion token to an asynchronous
 * operation, typically using the special value @c asio::deferred. For
 * example:
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

  /// Creates a new deferred from a function.
  template <typename Function>
  typename constraint<
    !is_deferred<typename decay<Function>::type>::value,
    deferred_function<typename decay<Function>::type>
  >::type operator()(ASIO_MOVE_ARG(Function) function) const
  {
    return deferred_function<typename decay<Function>::type>(
        deferred_init_tag{}, ASIO_MOVE_CAST(Function)(function));
  }

  /// Passes through anything that is already deferred.
  template <typename T>
  typename constraint<
    is_deferred<typename decay<T>::type>::value,
    typename decay<T>::type
  >::type operator()(ASIO_MOVE_ARG(T) t) const
  {
    return ASIO_MOVE_CAST(T)(t);
  }

  /// Returns a deferred operation that returns the provided values.
  template <typename... Args>
  static ASIO_CONSTEXPR deferred_values<typename decay<Args>::type...>
  values(ASIO_MOVE_ARG(Args)... args)
  {
    return deferred_values<typename decay<Args>::type...>(
        deferred_init_tag{}, ASIO_MOVE_CAST(Args)(args)...);
  }

  /// Creates a conditional object for branching deferred operations.
  static ASIO_CONSTEXPR deferred_conditional<> when(bool b)
  {
    return deferred_conditional<>(b);
  }
};

/// Pipe operator used to chain deferred operations.
template <typename Head, typename Tail>
inline auto operator|(Head head, ASIO_MOVE_ARG(Tail) tail)
  -> typename constraint<
      is_deferred<Head>::value,
      decltype(ASIO_MOVE_OR_LVALUE(Head)(head)(
            ASIO_MOVE_CAST(Tail)(tail)))
    >::type
{
  return ASIO_MOVE_OR_LVALUE(Head)(head)(
      ASIO_MOVE_CAST(Tail)(tail));
}

/// A special value, similar to std::nothrow.
/**
 * See the documentation for asio::experimental::deferred_t for a usage
 * example.
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
