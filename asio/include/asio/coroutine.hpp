//
// coroutine.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_COROUTINE_HPP
#define ASIO_COROUTINE_HPP

#include <asio/error_code.hpp>
#include <asio/system_error.hpp>

namespace asio {
namespace detail {

class coroutine_ref;

} // namespace detail

/// Provides support for implementing stackless coroutines.
/**
 * The @c coroutine class may be used to implement stackless coroutines. The
 * class itself is used to store the current state of the coroutine.
 *
 * Coroutines are copy-constructible and assignable, and the space overhead is
 * a single int. They can be used as a base class:
 *
 * @code class session : coroutine
 * {
 *   ...
 * }; @endcode
 *
 * or as a data member:
 *
 * @code class session
 * {
 *   ...
 *   coroutine coro_;
 * }; @endcode
 *
 * or even bound in as a function argument using lambdas or @c bind(). The
 * important thing is that as the application maintains a copy of the object
 * for as long as the coroutine must be kept alive.
 *
 * @par Pseudo-keywords
 *
 * A coroutine is used in conjunction with certain "pseudo-keywords", which
 * are implemented as macros. These macros are defined by a header file:
 *
 * @code #include <asio/yield.hpp>@endcode
 *
 * and may conversely be undefined as follows:
 *
 * @code #include <asio/unyield.hpp>@endcode
 *
 * <b>reenter</b>
 *
 * The @c reenter macro is used to define the body of a coroutine. It takes a
 * single argument: a pointer or reference to a coroutine object. For example,
 * if the base class is a coroutine object you may write:
 *
 * @code reenter (this)
 * {
 *   ... coroutine body ...
 * } @endcode
 *
 * and if a data member or other variable you can write:
 *
 * @code reenter (coro_)
 * {
 *   ... coroutine body ...
 * } @endcode
 *
 * When @c reenter is executed at runtime, control jumps to the location of the
 * last @c yield or @c fork.
 *
 * The coroutine body may also be a single statement, such as:
 *
 * @code reenter (this) for (;;)
 * {
 *   ...
 * } @endcode
 *
 * @b Limitation: The @c reenter macro is implemented using a switch. This
 * means that you must take care when using local variables within the
 * coroutine body. The local variable is not allowed in a position where
 * reentering the coroutine could bypass the variable definition.
 *
 * <b>yield <em>statement</em></b>
 *
 * This form of the @c yield keyword is often used with asynchronous operations:
 *
 * @code yield socket_->async_read_some(buffer(*buffer_), *this); @endcode
 *
 * This divides into four logical steps:
 *
 * @li @c yield saves the current state of the coroutine.
 * @li The statement initiates the asynchronous operation.
 * @li The resume point is defined immediately following the statement.
 * @li Control is transferred to the end of the coroutine body.
 *
 * When the asynchronous operation completes, the function object is invoked
 * and @c reenter causes control to transfer to the resume point. It is
 * important to remember to carry the coroutine state forward with the
 * asynchronous operation. In the above snippet, the current class is a
 * function object object with a coroutine object as base class or data member.
 *
 * The statement may also be a compound statement, and this permits us to
 * define local variables with limited scope:
 *
 * @code yield
 * {
 *   mutable_buffers_1 b = buffer(*buffer_);
 *   socket_->async_read_some(b, *this);
 * } @endcode
 *
 * <b>yield return <em>expression</em> ;</b>
 *
 * This form of @c yield is often used in generators or coroutine-based parsers.
 * For example, the function object:
 *
 * @code struct interleave : coroutine
 * {
 *   istream& is1;
 *   istream& is2;
 *   char operator()(char c)
 *   {
 *     reenter (this) for (;;)
 *     {
 *       yield return is1.get();
 *       yield return is2.get();
 *     }
 *   }
 * }; @endcode
 *
 * defines a trivial coroutine that interleaves the characters from two input
 * streams.
 *
 * This type of @c yield divides into three logical steps:
 *
 * @li @c yield saves the current state of the coroutine.
 * @li The resume point is defined immediately following the semicolon.
 * @li The value of the expression is returned from the function.
 *
 * <b>yield ;</b>
 *
 * This form of @c yield is equivalent to the following steps:
 *
 * @li @c yield saves the current state of the coroutine.
 * @li The resume point is defined immediately following the semicolon.
 * @li Control is transferred to the end of the coroutine body.
 *
 * This form might be applied when coroutines are used for cooperative
 * threading and scheduling is explicitly managed. For example:
 *
 * @code struct task : coroutine
 * {
 *   ...
 *   void operator()()
 *   {
 *     reenter (this)
 *     {
 *       while (... not finished ...)
 *       {
 *         ... do something ...
 *         yield;
 *         ... do some more ...
 *         yield;
 *       }
 *     }
 *   }
 *   ...
 * };
 * ...
 * task t1, t2;
 * for (;;)
 * {
 *   t1();
 *   t2();
 * } @endcode
 *
 * <b>yield break ;</b>
 *
 * The final form of @c yield is used to explicitly terminate the coroutine.
 * This form is comprised of two steps:
 *
 * @li @c yield sets the coroutine state to indicate termination.
 * @li Control is transferred to the end of the coroutine body.
 *
 * Once terminated, calls to is_complete() return true and the coroutine cannot
 * be reentered.
 *
 * Note that a coroutine may also be implicitly terminated if the coroutine
 * body is exited without a yield, e.g. by return, throw or by running to the
 * end of the body.
 *
 * <b>fork <em>statement</em></b>
 *
 * The @c fork pseudo-keyword is used when "forking" a coroutine, i.e. splitting
 * it into two (or more) copies. One use of @c fork is in a server, where a new
 * coroutine is created to handle each client connection:
 * 
 * @code reenter (this)
 * {
 *   do
 *   {
 *     socket_.reset(new tcp::socket(io_service_));
 *     yield acceptor->async_accept(*socket_, *this);
 *     fork server(*this)();
 *   } while (is_parent());
 *   ... client-specific handling follows ...
 * } @endcode
 * 
 * The logical steps involved in a @c fork are:
 * 
 * @li @c fork saves the current state of the coroutine.
 * @li The statement creates a copy of the coroutine and either executes it
 *     immediately or schedules it for later execution.
 * @li The resume point is defined immediately following the semicolon.
 * @li For the "parent", control immediately continues from the next line.
 *
 * The functions is_parent() and is_child() can be used to differentiate
 * between parent and child. You would use these functions to alter subsequent
 * control flow.
 *
 * Note that @c fork doesn't do the actual forking by itself. It is the
 * application's responsibility to create a clone of the coroutine and call it.
 * The clone can be called immediately, as above, or scheduled for delayed
 * execution using something like io_service::post().
 *
 * @par Alternate macro names
 *
 * If preferred, an application can use macro names that follow a more typical
 * naming convention, rather than the pseudo-keywords. These are:
 *
 * @li @c ASIO_CORO_REENTER instead of @c reenter
 * @li @c ASIO_CORO_YIELD instead of @c yield
 * @li @c ASIO_CORO_FORK instead of @c fork
 * @li @c ASIO_CORO_LET instead of @c let
 * @li @c ASIO_CORO_AWAIT instead of @c await
 */
class coroutine
{
public:
  /// Constructs a coroutine in its initial state.
  coroutine() : value_(0) {}

  /// Returns true if the coroutine is the child of a fork.
  bool is_child() const { return value_ < 0; }

  /// Returns true if the coroutine is the parent of a fork.
  bool is_parent() const { return !is_child(); }

  /// Returns true if the coroutine has reached its terminal state.
  bool is_complete() const { return value_ == -1; }

  /// Used by the @c reenter pseudo-keyword to obtain the coroutine state.
  friend int& coroutine_state(coroutine& c) { return c.value_; }

  /// Used by the @c reenter pseudo-keyword to obtain the coroutine state.
  friend int& coroutine_state(coroutine* c) { return c->value_; }

  /// Used by the @c reenter pseudo-keyword to obtain the error code resulting
  /// from the previous operation. If set, an exception will be thrown
  /// immediately following the resumption point.
  friend const asio::error_code* coroutine_error(coroutine&) { return 0; }

  /// Used by the @c reenter pseudo-keyword to obtain the error code resulting
  /// from the previous operation. If set, an exception will be thrown
  /// immediately following the resumption point.
  friend const asio::error_code* coroutine_error(coroutine*) { return 0; }

  /// Called by the @c let and @c await pseudo-keywords to obtain the pointer
  /// used to refer to any variables that should be set from the result of an
  /// asynchronous operation.
  friend void** coroutine_async_result(coroutine&) { return 0; }

  /// Called by the @c let and @c await pseudo-keywords to obtain the pointer
  /// used to refer to any variables that should be set from the result of an
  /// asynchronous operation.
  friend void** coroutine_async_result(coroutine*) { return 0; }

private:
  int value_;
};

namespace detail {

template <typename T> class coroutine_async_result {};

class coroutine_ref
{
public:
  // Construct a coroutine reference for use in the pseudo-keywords.
  coroutine_ref(int& value, const asio::error_code* ec, void** result)
    : value_(value), ec_(ec), async_result_(result), modified_(false) {}

  // Destructor sets coroutine to the completed state unless explicitly set.
  ~coroutine_ref() { if (!modified_) value_ = -1; }

  // Obtain the coroutine state.
  operator int() const { return value_; }

  // Set the coroutine state.
  int& operator=(int v) { modified_ = true; return value_ = v; }

  // Operator used to associate a variable to store the async result.
  template <typename T>
  coroutine_async_result<T> operator&(T& t)
  {
    *async_result_ = &t;
    return coroutine_async_result<T>();
  }

  // This overload is used when the result is ignored. 
  template <typename T> void operator&(coroutine_async_result<T>) {}

  // Throw an exception if the coroutine has an associated error.
  void throw_on_error() const
  {
    if (ec_ && *ec_) throw asio::system_error(*ec_);
  }

private:
  void operator=(const coroutine_ref&);
  int& value_;
  const asio::error_code* ec_;
  void** async_result_;
  bool modified_;
};

} // namespace detail
} // namespace asio

#define ASIO_CORO_REENTER(c) \
  switch (::asio::detail::coroutine_ref _coro_value = \
      ::asio::detail::coroutine_ref(coroutine_state(c), \
        coroutine_error(c), coroutine_async_result(c))) \
    case -1: if (_coro_value) \
    { \
      goto terminate_coroutine; \
      terminate_coroutine: \
      _coro_value = -1; \
      goto bail_out_of_coroutine; \
      bail_out_of_coroutine: \
      break; \
    } \
    else case 0:

#define ASIO_CORO_YIELD_IMPL(n) \
  for (_coro_value = (n);;) \
    if (_coro_value == 0) \
    { \
      case (n): ; \
      _coro_value.throw_on_error(); \
      break; \
    } \
    else \
      switch (_coro_value ? 0 : 1) \
        for (;;) \
          case -1: if (_coro_value) \
            goto terminate_coroutine; \
          else for (;;) \
            case 1: if (_coro_value) \
              goto bail_out_of_coroutine; \
            else case 0:

#define ASIO_CORO_FORK_IMPL(n) \
  for (_coro_value = -(n);; _coro_value = (n)) \
    if (_coro_value == (n)) \
    { \
      case -(n): ; \
      break; \
    } \
    else

#if defined(_MSC_VER)
# define ASIO_CORO_YIELD ASIO_CORO_YIELD_IMPL(__COUNTER__ + 1)
# define ASIO_CORO_FORK ASIO_CORO_FORK_IMPL(__COUNTER__ + 1)
#else // defined(_MSC_VER)
# define ASIO_CORO_YIELD ASIO_CORO_YIELD_IMPL(__LINE__)
# define ASIO_CORO_FORK ASIO_CORO_FORK_IMPL(__LINE__)
#endif // defined(_MSC_VER)

#define ASIO_CORO_LET _coro_value&

#define ASIO_CORO_AWAIT ASIO_CORO_YIELD ASIO_CORO_LET

#endif // ASIO_COROUTINE_HPP
