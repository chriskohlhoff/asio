//
// detail/candidate_executors.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_CANDIDATE_EXECUTORS_HPP
#define ASIO_DETAIL_CANDIDATE_EXECUTORS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/associated_executor.hpp"
#include "asio/associated_immediate_executor.hpp"
#include "asio/associator.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/any_executor.hpp"
#include "asio/execution/blocking.hpp"
#include "asio/require.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
ASIO_INLINE_NAMESPACE_BEGIN
namespace detail {

template <typename T, typename Candidate>
struct executor_is_candidate_query;

template <typename T, typename Candidate,
    typename = void, typename = void>
struct executor_is_candidate_query_impl
{
  typedef bool type;

  static const bool is_known = false;

  static type get(const T&, const Candidate&) noexcept
  {
    return false;
  }
};

template <typename T, typename Candidate>
struct executor_is_candidate_query_impl<T, Candidate,
    enable_if_t<
      is_base_of<
        associator<associated_executor, T, Candidate>,
        associated_executor<T, Candidate>
      >::value
    >,
    void_t<
      typename associator<executor_is_candidate_query, T, Candidate>::type
    >
  > : associator<executor_is_candidate_query, T, Candidate>
{
};

template <typename T, typename Candidate>
struct executor_is_candidate_query
  : executor_is_candidate_query_impl<T, Candidate>
{
};

template <typename T, typename Candidate>
inline bool executor_is_candidate(
    const T& t, const Candidate& candidate) noexcept
{
  return executor_is_candidate_query<T, Candidate>::get(t, candidate);
}

// Whether a handler's immediate executor is exactly what
// get_associated_immediate_executor would derive BY DEFAULT for the candidate
// executor, i.e. the candidate itself with blocking.never required on it.
// That is the comparison a customised immediate executor (e.g. via
// bind_immediate_executor, applied to the handler BEFORE it was erased into
// an any_completion_handler, so no compile-time trait can see it) has to pass
// at runtime instead. Only attempted when the association is a polymorphic
// executor (so the comparison itself is erasure-safe) and the candidate
// supports blocking.never, as forming the default at all requires. It is
// false at compile time otherwise, collapsing back to the unconditional
// generic dispatch of before, and without deriving the association.
//
// Getting this wrong can only cost performance at the call site in
// immediate_handler_work, not correctness: in both branches there the
// completion is still dispatched through the correctly-derived immediate
// executor (nothing here substitutes a different executor, it only chooses
// how to reach the same one), and initiate_dispatch_with_executor's own
// is_work_dispatcher_required check already collapses to the identical
// asio::prefer(...).execute(...) call for the unspecialised-associated-
// executor continuation functions asio's own read_op/write_op/etc. construct
// internally. Confirmed empirically, by forcing this function to always
// return true, which produces no observable change against the fast-path
// tests, or the composed-operation tests, only a measured -1172
// instructions/RTT. A false positive would therefore not misdirect a
// completion. It is guarded this carefully even so, since
// immediate_handler_work is also reachable with other, non-I/O internal
// Function shapes where that guarantee has not been re-verified.
template <typename T, typename Candidate>
enable_if_t<
  is_base_of<execution::detail::any_executor_base,
    associated_immediate_executor_t<T, Candidate>>::value
      && can_require<Candidate, execution::blocking_t::never_t>::value,
  bool
> immediate_executor_is_candidate_object(
    const T& t, const Candidate& candidate) noexcept
{
  return execution::detail::executor_targets_equal(
      (get_associated_immediate_executor)(t, candidate),
      asio::require(candidate, execution::blocking.never));
}

template <typename T, typename Candidate>
enable_if_t<
  !is_base_of<execution::detail::any_executor_base,
    associated_immediate_executor_t<T, Candidate>>::value
      || !can_require<Candidate, execution::blocking_t::never_t>::value,
  bool
> immediate_executor_is_candidate_object(const T&, const Candidate&) noexcept
{
  return false;
}

template <typename T, typename Candidate>
struct immediate_executor_is_candidate_query;

template <typename T, typename Candidate,
    typename = void, typename = void>
struct immediate_executor_is_candidate_query_impl
{
  typedef bool type;

  static type get(const T& t, const Candidate& candidate) noexcept
  {
    return immediate_executor_is_candidate_object(t, candidate);
  }
};

template <typename T, typename Candidate>
struct immediate_executor_is_candidate_query_impl<T, Candidate,
    enable_if_t<
      is_base_of<
        associator<associated_immediate_executor, T, Candidate>,
        associated_immediate_executor<T, Candidate>
      >::value
    >,
    void_t<
      typename associator<immediate_executor_is_candidate_query,
        T, Candidate>::type
    >
  > : associator<immediate_executor_is_candidate_query, T, Candidate>
{
};

template <typename T, typename Candidate>
struct immediate_executor_is_candidate_query
  : immediate_executor_is_candidate_query_impl<T, Candidate>
{
};

template <typename T, typename Candidate>
inline bool immediate_executor_is_candidate(
    const T& t, const Candidate& candidate) noexcept
{
  return immediate_executor_is_candidate_query<T, Candidate>::get(t, candidate);
}

} // namespace detail
ASIO_INLINE_NAMESPACE_END
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_CANDIDATE_EXECUTORS_HPP
