//
// experimental/diagnostic_executor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2026 Pinwhell <binarydetective@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_DIAGNOSTIC_EXECUTOR_HPP
#define ASIO_EXPERIMENTAL_DIAGNOSTIC_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/blocking.hpp"
#include "asio/execution/executor.hpp"
#include "asio/execution_context.hpp"
#include "asio/query.hpp"
#include "asio/require.hpp"
#include "asio/prefer.hpp"
#include "asio/dispatch.hpp"
#include "asio/post.hpp"
#include "asio/defer.hpp"
#include "asio/bind_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {

/// Default diagnostic policy that performs no action.
/**
 * This policy provides a no-op implementation of the diagnostic hooks, ensuring
 * that the diagnostic executor has zero overhead when no diagnostics are
 * required.
 */
struct null_diagnostic_policy
{
  /// Hook called before work is submitted to the underlying executor.
  template <typename Label>
  static void on_submit(const Label&) noexcept
  {
  }
};

/// An executor adapter that transparently observes work submission.
/**
 * The diagnostic_executor class template is used to wrap another executor and
 * provide a point of observation for work submission. It forwards all
 * operations to the underlying executor while invoking a diagnostic policy
 * hook whenever work is submitted via @c execute, @c dispatch, @c post, or
 * @c defer.
 */
template <typename InnerExecutor, 
          typename Label = const char*, 
          typename DiagnosticPolicy = null_diagnostic_policy>
class diagnostic_executor
{
public:
  /// The type of the underlying executor.
  typedef InnerExecutor inner_executor_type;

  /// The type of the underlying executor.
  typedef InnerExecutor nested_executor_type;

  /// The type of the diagnostic label.
  typedef Label label_type;

  /// The type of the diagnostic policy.
  typedef DiagnosticPolicy diagnostic_policy_type;

  /// Construct from an inner executor and a label.
  template <typename InnerEx, typename Lbl>
  diagnostic_executor(ASIO_MOVE_ARG(InnerEx) inner, ASIO_MOVE_ARG(Lbl) label)
    : inner_(ASIO_MOVE_CAST(InnerEx)(inner)),
      label_(ASIO_MOVE_CAST(Lbl)(label))
  {
  }

  /// Copy constructor.
  diagnostic_executor(const diagnostic_executor& other) noexcept
    : inner_(other.inner_),
      label_(other.label_)
  {
  }

  /// Move constructor.
  diagnostic_executor(diagnostic_executor&& other) noexcept
    : inner_(ASIO_MOVE_CAST(InnerExecutor)(other.inner_)),
      label_(ASIO_MOVE_CAST(Label)(other.label_))
  {
  }

  /// Assignment operator.
  diagnostic_executor& operator=(const diagnostic_executor& other) noexcept
  {
    inner_ = other.inner_;
    label_ = other.label_;
    return *this;
  }

  /// Move assignment operator.
  diagnostic_executor& operator=(diagnostic_executor&& other) noexcept
  {
    inner_ = ASIO_MOVE_CAST(InnerExecutor)(other.inner_);
    label_ = ASIO_MOVE_CAST(Label)(other.label_);
    return *this;
  }

  /// Compare two executors for equality.
  /**
   * Two diagnostic executors are equal if their underlying executors are equal
   * and their labels are equal.
   */
  friend bool operator==(const diagnostic_executor& a,
      const diagnostic_executor& b) noexcept
  {
    return a.inner_ == b.inner_ && a.label_ == b.label_;
  }

  /// Compare two executors for inequality.
  friend bool operator!=(const diagnostic_executor& a,
      const diagnostic_executor& b) noexcept
  {
    return !(a == b);
  }

  /// Execution function to submit a function object for execution.
  /**
   * Invokes the diagnostic policy's @c on_submit hook before forwarding the
   * function object to the underlying executor's @c execute function.
   */
  template <typename Function>
  void execute(ASIO_MOVE_ARG(Function) f) const
  {
    DiagnosticPolicy::on_submit(label_);
    inner_.execute(ASIO_MOVE_CAST(Function)(f));
  }

#if !defined(ASIO_NO_TS_EXECUTORS)
  /// Obtain the underlying execution context.
  execution_context& context() const noexcept
  {
    return inner_.context();
  }

  /// Inform the executor that it has some outstanding work to do.
  void on_work_started() const noexcept
  {
    inner_.on_work_started();
  }

  /// Inform the executor that some work is no longer outstanding.
  void on_work_finished() const noexcept
  {
    inner_.on_work_finished();
  }

  /// Request the underlying executor to invoke the given function object.
  template <typename Function>
  void dispatch(ASIO_MOVE_ARG(Function) f) const
  {
    DiagnosticPolicy::on_submit(label_);
    asio::dispatch(inner_, ASIO_MOVE_CAST(Function)(f));
  }

  /// Request the underlying executor to invoke the given function object.
  template <typename Function, typename Allocator>
  void dispatch(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
  {
    DiagnosticPolicy::on_submit(label_);
    inner_.dispatch(asio::bind_executor(*this, ASIO_MOVE_CAST(Function)(f)), a);
  }

  /// Request the underlying executor to invoke the given function object.
  template <typename Function>
  void post(ASIO_MOVE_ARG(Function) f) const
  {
    DiagnosticPolicy::on_submit(label_);
    asio::post(inner_, ASIO_MOVE_CAST(Function)(f));
  }

  /// Request the underlying executor to invoke the given function object.
  template <typename Function, typename Allocator>
  void post(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
  {
    DiagnosticPolicy::on_submit(label_);
    inner_.post(asio::bind_executor(*this, ASIO_MOVE_CAST(Function)(f)), a);
  }

  /// Request the underlying executor to invoke the given function object.
  template <typename Function>
  void defer(ASIO_MOVE_ARG(Function) f) const
  {
    DiagnosticPolicy::on_submit(label_);
    asio::defer(inner_, ASIO_MOVE_CAST(Function)(f));
  }

  /// Request the underlying executor to invoke the given function object.
  template <typename Function, typename Allocator>
  void defer(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
  {
    DiagnosticPolicy::on_submit(label_);
    inner_.defer(asio::bind_executor(*this, ASIO_MOVE_CAST(Function)(f)), a);
  }
#endif // !defined(ASIO_NO_TS_EXECUTORS)

  /// Get the underlying executor.
  ASIO_NODISCARD const InnerExecutor& get_inner_executor() const noexcept
  {
    return inner_;
  }

  /// Forward a query to the underlying executor.
  template <typename Property>
  ASIO_NODISCARD auto query(ASIO_MOVE_ARG(Property) p) const 
    noexcept(asio::can_query<const InnerExecutor&, Property>::value && 
             asio::is_nothrow_query<const InnerExecutor&, Property>::value)
    -> asio::query_result_t<const InnerExecutor&, Property>
  {
    return asio::query(inner_, ASIO_MOVE_CAST(Property)(p));
  }

  /// Forward a requirement to the underlying executor.
  template <typename Property>
  ASIO_NODISCARD auto require(ASIO_MOVE_ARG(Property) p) const 
    noexcept(asio::can_require<const InnerExecutor&, Property>::value && 
             asio::is_nothrow_require<const InnerExecutor&, Property>::value)
    -> diagnostic_executor<asio::decay_t<asio::require_result_t<const InnerExecutor&, Property>>, Label, DiagnosticPolicy>
  {
    return diagnostic_executor<asio::decay_t<asio::require_result_t<const InnerExecutor&, Property>>, Label, DiagnosticPolicy>(
        asio::require(inner_, ASIO_MOVE_CAST(Property)(p)), label_);
  }

  /// Forward a preference to the underlying executor.
  template <typename Property>
  ASIO_NODISCARD auto prefer(ASIO_MOVE_ARG(Property) p) const 
    noexcept(asio::can_prefer<const InnerExecutor&, Property>::value && 
             asio::is_nothrow_prefer<const InnerExecutor&, Property>::value)
    -> diagnostic_executor<asio::decay_t<asio::prefer_result_t<const InnerExecutor&, Property>>, Label, DiagnosticPolicy>
  {
    return diagnostic_executor<asio::decay_t<asio::prefer_result_t<const InnerExecutor&, Property>>, Label, DiagnosticPolicy>(
        asio::prefer(inner_, ASIO_MOVE_CAST(Property)(p)), label_);
  }

private:
  InnerExecutor inner_;
  Label label_;
};

/// Create a diagnostic executor for the specified executor and label.
template <typename Executor, typename Label>
ASIO_NODISCARD inline diagnostic_executor<asio::decay_t<Executor>, asio::decay_t<Label>>
make_diagnostic_executor(ASIO_MOVE_ARG(Executor) ex, ASIO_MOVE_ARG(Label) label)
{
  return diagnostic_executor<asio::decay_t<Executor>, asio::decay_t<Label>>(
      ASIO_MOVE_CAST(Executor)(ex), ASIO_MOVE_CAST(Label)(label));
}

} // namespace experimental

namespace traits {

#if !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)
template <typename InnerExecutor, typename Label, typename DiagnosticPolicy>
struct equality_comparable<experimental::diagnostic_executor<InnerExecutor, Label, DiagnosticPolicy>>
{
  static constexpr bool is_valid = asio::traits::equality_comparable<InnerExecutor>::is_valid;
  static constexpr bool is_noexcept = asio::traits::equality_comparable<InnerExecutor>::is_noexcept;
};
#endif // !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)
template <typename InnerExecutor, typename Label, typename DiagnosticPolicy, typename Function>
struct execute_member<experimental::diagnostic_executor<InnerExecutor, Label, DiagnosticPolicy>, Function,
    asio::enable_if_t<
      asio::traits::execute_member<const InnerExecutor&, Function>::is_valid
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = asio::traits::execute_member<const InnerExecutor&, Function>::is_noexcept;
  typedef void result_type;
};
#endif // !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT)
template <typename InnerExecutor, typename Label, typename DiagnosticPolicy, typename Property>
struct static_query<experimental::diagnostic_executor<InnerExecutor, Label, DiagnosticPolicy>, Property,
    asio::enable_if_t<
      asio::traits::static_query<InnerExecutor, Property>::is_valid
    >>
{
  static constexpr bool is_valid = true;
  typedef typename asio::traits::static_query<InnerExecutor, Property>::result_type result_type;
  static constexpr bool is_noexcept = asio::traits::static_query<InnerExecutor, Property>::is_noexcept;

  static constexpr result_type value() noexcept(is_noexcept)
  {
    return asio::traits::static_query<InnerExecutor, Property>::value();
  }
};
#endif // !defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)
template <typename InnerExecutor, typename Label, typename DiagnosticPolicy, typename Property>
struct query_member<experimental::diagnostic_executor<InnerExecutor, Label, DiagnosticPolicy>, Property,
    asio::enable_if_t<
      asio::can_query<const InnerExecutor&, Property>::value
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = asio::is_nothrow_query<const InnerExecutor&, Property>::value;
  typedef asio::query_result_t<const InnerExecutor&, Property> result_type;
};
#endif // !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)
template <typename InnerExecutor, typename Label, typename DiagnosticPolicy, typename Property>
struct require_member<experimental::diagnostic_executor<InnerExecutor, Label, DiagnosticPolicy>, Property,
    asio::enable_if_t<
      asio::can_require<const InnerExecutor&, Property>::value
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = asio::is_nothrow_require<const InnerExecutor&, Property>::value;
  typedef experimental::diagnostic_executor<
    asio::decay_t<asio::require_result_t<const InnerExecutor&, Property>>,
    Label, DiagnosticPolicy> result_type;
};
#endif // !defined(ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_PREFER_MEMBER_TRAIT)
template <typename InnerExecutor, typename Label, typename DiagnosticPolicy, typename Property>
struct prefer_member<experimental::diagnostic_executor<InnerExecutor, Label, DiagnosticPolicy>, Property,
    asio::enable_if_t<
      asio::can_prefer<const InnerExecutor&, Property>::value
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = asio::is_nothrow_prefer<const InnerExecutor&, Property>::value;
  typedef experimental::diagnostic_executor<
    asio::decay_t<asio::prefer_result_t<const InnerExecutor&, Property>>,
    Label, DiagnosticPolicy> result_type;
};
#endif // !defined(ASIO_HAS_DEDUCED_PREFER_MEMBER_TRAIT)

} // namespace traits
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_DIAGNOSTIC_EXECUTOR_HPP
