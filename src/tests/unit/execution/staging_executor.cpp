//
// staging_executor.cpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/execution/staging_executor.hpp"

#include <cstring>
#include <memory>
#include <typeinfo>
#include <utility>
#include "asio/any_io_executor.hpp"
#include "asio/detail/executor_function.hpp"
#include "asio/execution/outstanding_work.hpp"
#include "asio/io_context.hpp"
#include "asio/strand.hpp"
#include "asio/thread_pool.hpp"
#include "../unit_test.hpp"

using namespace asio;

namespace {

struct test_log
{
  static int copies;
  static int moves;
  static char order[16];
  static int order_len;

  static void reset()
  {
    copies = 0;
    moves = 0;
    order_len = 0;
    order[0] = 0;
  }

  static void applied(char c)
  {
    if (order_len < 15)
    {
      order[order_len++] = c;
      order[order_len] = 0;
    }
  }
};

int test_log::copies = 0;
int test_log::moves = 0;
char test_log::order[16] = "";
int test_log::order_len = 0;

template <typename T>
struct test_allocator
{
  typedef T value_type;

  explicit test_allocator(int id) noexcept
    : id_(id)
  {
  }

  template <typename U>
  test_allocator(const test_allocator<U>& other) noexcept
    : id_(other.id_)
  {
  }

  T* allocate(std::size_t n)
  {
    return static_cast<T*>(::operator new(sizeof(T) * n));
  }

  void deallocate(T* p, std::size_t)
  {
    ::operator delete(p);
  }

  template <typename U>
  struct rebind
  {
    typedef test_allocator<U> other;
  };

  friend bool operator==(const test_allocator& a,
      const test_allocator& b) noexcept
  {
    return a.id_ == b.id_;
  }

  friend bool operator!=(const test_allocator& a,
      const test_allocator& b) noexcept
  {
    return a.id_ != b.id_;
  }

  int id_;
};

template <typename Blocking = execution::blocking_t::possibly_t,
    typename Relationship = execution::relationship_t::fork_t>
class test_executor
{
public:
  explicit test_executor(io_context& ctx, int id = 0) noexcept
    : context_(&ctx),
      id_(id)
  {
  }

  test_executor(const test_executor& other) noexcept
    : context_(other.context_),
      id_(other.id_)
  {
    ++test_log::copies;
  }

  test_executor(test_executor&& other) noexcept
    : context_(other.context_),
      id_(other.id_)
  {
    ++test_log::moves;
  }

  test_executor<execution::blocking_t::possibly_t, Relationship>
  require(const execution::blocking_t::possibly_t&) const
  {
    test_log::applied('b');
    return test_executor<execution::blocking_t::possibly_t,
        Relationship>(*context_, id_);
  }

  test_executor<execution::blocking_t::never_t, Relationship>
  require(const execution::blocking_t::never_t&) const
  {
    test_log::applied('b');
    return test_executor<execution::blocking_t::never_t,
        Relationship>(*context_, id_);
  }

  test_executor<Blocking, execution::relationship_t::fork_t>
  require(const execution::relationship_t::fork_t&) const
  {
    test_log::applied('r');
    return test_executor<Blocking,
        execution::relationship_t::fork_t>(*context_, id_);
  }

  test_executor<Blocking, execution::relationship_t::continuation_t>
  require(const execution::relationship_t::continuation_t&) const
  {
    test_log::applied('r');
    return test_executor<Blocking,
        execution::relationship_t::continuation_t>(*context_, id_);
  }

  template <typename OtherAllocator>
  test_executor require(
      const execution::allocator_t<OtherAllocator>&) const
  {
    test_log::applied('a');
    return test_executor(*context_, id_, 0);
  }

  static constexpr execution::blocking_t query(
      execution::blocking_t) noexcept
  {
    return Blocking();
  }

  static constexpr execution::relationship_t query(
      execution::relationship_t) noexcept
  {
    return Relationship();
  }

  execution_context& query(execution::context_t) const noexcept
  {
    return *context_;
  }

  template <typename Function>
  void execute(Function&& f) const
  {
    asio::require(context_->get_executor(),
        execution::blocking.never).execute(static_cast<Function&&>(f));
  }

  friend bool operator==(const test_executor& a,
      const test_executor& b) noexcept
  {
    return a.context_ == b.context_ && a.id_ == b.id_;
  }

  friend bool operator!=(const test_executor& a,
      const test_executor& b) noexcept
  {
    return !(a == b);
  }

private:
  template <typename, typename> friend class test_executor;

  test_executor(io_context& ctx, int id, int) noexcept
    : context_(&ctx),
      id_(id)
  {
  }

  io_context* context_;
  int id_;
};

struct bare_executor
{
  template <typename Function>
  void execute(Function&& f) const
  {
    static_cast<Function&&>(f)();
  }

  friend bool operator==(const bare_executor&,
      const bare_executor&) noexcept
  {
    return true;
  }

  friend bool operator!=(const bare_executor&,
      const bare_executor&) noexcept
  {
    return false;
  }
};

typedef test_executor<> test_ex;
typedef execution::blocking_t::never_t never_t;
typedef execution::blocking_t::possibly_t possibly_t;
typedef execution::relationship_t::fork_t fork_t;
typedef execution::relationship_t::continuation_t continuation_t;
typedef execution::outstanding_work_t::tracked_t tracked_t;

struct increment_function
{
  int* count;

  void operator()()
  {
    ++(*count);
  }
};

} // namespace

namespace asio {
namespace traits {

#if !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

template <typename Blocking, typename Relationship>
struct equality_comparable<test_executor<Blocking, Relationship>>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
};

template <>
struct equality_comparable<bare_executor>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
};

#endif // !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

template <typename Blocking, typename Relationship, typename Function>
struct execute_member<test_executor<Blocking, Relationship>, Function>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef void result_type;
};

template <typename Function>
struct execute_member<bare_executor, Function>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_QUERY_STATIC_CONSTEXPR_MEMBER_TRAIT)

template <typename Blocking, typename Relationship, typename Property>
struct query_static_constexpr_member<test_executor<Blocking, Relationship>,
    Property,
    enable_if_t<
      is_convertible<Property, execution::blocking_t>::value
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
  typedef Blocking result_type;

  static constexpr result_type value() noexcept
  {
    return result_type();
  }
};

template <typename Blocking, typename Relationship, typename Property>
struct query_static_constexpr_member<test_executor<Blocking, Relationship>,
    Property,
    enable_if_t<
      is_convertible<Property, execution::relationship_t>::value
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
  typedef Relationship result_type;

  static constexpr result_type value() noexcept
  {
    return result_type();
  }
};

#endif // !defined(ASIO_HAS_DEDUCED_QUERY_STATIC_CONSTEXPR_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

template <typename Blocking, typename Relationship>
struct query_member<test_executor<Blocking, Relationship>,
    execution::context_t>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
  typedef execution_context& result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)

template <typename Blocking, typename Relationship>
struct require_member<test_executor<Blocking, Relationship>,
    execution::blocking_t::possibly_t>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef test_executor<execution::blocking_t::possibly_t,
      Relationship> result_type;
};

template <typename Blocking, typename Relationship>
struct require_member<test_executor<Blocking, Relationship>,
    execution::blocking_t::never_t>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef test_executor<execution::blocking_t::never_t,
      Relationship> result_type;
};

template <typename Blocking, typename Relationship>
struct require_member<test_executor<Blocking, Relationship>,
    execution::relationship_t::fork_t>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef test_executor<Blocking,
      execution::relationship_t::fork_t> result_type;
};

template <typename Blocking, typename Relationship>
struct require_member<test_executor<Blocking, Relationship>,
    execution::relationship_t::continuation_t>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef test_executor<Blocking,
      execution::relationship_t::continuation_t> result_type;
};

template <typename Blocking, typename Relationship, typename OtherAllocator>
struct require_member<test_executor<Blocking, Relationship>,
    execution::allocator_t<OtherAllocator>>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef test_executor<Blocking, Relationship> result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)

} // namespace traits
} // namespace asio

static_assert(
    execution::is_executor<
      execution::staging_executor<const test_ex&>
    >::value, "staging_executor");

static_assert(
    execution::is_executor<
      execution::staging_executor<test_ex>
    >::value, "staging_executor");

static_assert(
    execution::is_executor<
      execution::staging_executor<const test_ex&, never_t,
        execution::prefer_only<continuation_t>,
        execution::allocator_t<std::allocator<void>>>
    >::value, "staging_executor");

static_assert(
    execution::is_executor<
      execution::staging_executor<const any_io_executor&, never_t>
    >::value, "staging_executor");

static_assert(
    execution::is_executor<
      execution::staging_executor<any_io_executor, never_t,
        execution::prefer_only<fork_t>>
    >::value, "staging_executor");

static_assert(
    execution::detail::staging_execution_kind_of<
      const any_io_executor&, never_t
    >::value == execution::detail::staged_execution_kind::never,
    "staging_executor");

static_assert(
    execution::detail::staging_execution_kind_of<
      const any_io_executor&, never_t, execution::prefer_only<fork_t>
    >::value == execution::detail::staged_execution_kind::never_fork,
    "staging_executor");

static_assert(
    execution::detail::staging_execution_kind_of<
      any_io_executor, never_t, execution::prefer_only<continuation_t>
    >::value == execution::detail::staged_execution_kind::never_continuation,
    "staging_executor");

static_assert(
    execution::detail::staging_execution_kind_of<
      const any_io_executor&, possibly_t
    >::value == execution::detail::staged_execution_kind::none,
    "staging_executor");

static_assert(
    execution::detail::staging_execution_kind_of<
      const any_io_executor&, execution::prefer_only<fork_t>
    >::value == execution::detail::staged_execution_kind::none,
    "staging_executor");

static_assert(
    execution::detail::staging_execution_kind_of<
      const any_io_executor&, never_t,
      execution::prefer_only<execution::allocator_t<std::allocator<void>>>
    >::value == execution::detail::staged_execution_kind::none,
    "staging_executor");

static_assert(
    execution::detail::staging_execution_kind_of<
      const test_ex&, never_t
    >::value == execution::detail::staged_execution_kind::none,
    "staging_executor");

static_assert(
    execution::detail::staging_execution_kind_of<
      const any_io_executor&, never_t, never_t
    >::value == execution::detail::staged_execution_kind::none,
    "staging_executor");

static_assert(
    execution::detail::staging_execution_kind_of<
      const any_io_executor&, execution::prefer_only<fork_t>, never_t
    >::value == execution::detail::staged_execution_kind::never_fork,
    "staging_executor");

static_assert(
    !execution::detail::is_valid_target_executor<
      execution::staging_executor<const test_ex&>,
      any_io_executor::supportable_properties_type
    >::value, "staging_executor");

void staging_executor_ownership_test()
{
  io_context ctx;
  test_ex ex(ctx, 1);

  test_log::reset();
  execution::staging_executor<const test_ex&> s1 = execution::stage(ex);
  ASIO_CHECK(test_log::copies == 0);
  ASIO_CHECK(test_log::moves == 0);
  ASIO_CHECK(&s1.get_inner_executor() == &ex);

  execution::staging_executor<const test_ex&> s2(s1);
  ASIO_CHECK(&s2.get_inner_executor() == &ex);
  ASIO_CHECK(test_log::copies == 0);

  execution::staging_executor<const test_ex&, never_t> s3 =
    asio::require(s1, execution::blocking.never);
  ASIO_CHECK(&s3.get_inner_executor() == &ex);
  ASIO_CHECK(test_log::copies == 0);
  ASIO_CHECK(test_log::moves == 0);

  test_log::reset();
  execution::staging_executor<test_ex> s4 =
    execution::stage(test_ex(ctx, 2));
  ASIO_CHECK(test_log::copies == 0);
  ASIO_CHECK(test_log::moves >= 1);
  ASIO_CHECK(s4.get_inner_executor() == test_ex(ctx, 2));

  test_log::reset();
  execution::staging_executor<test_ex, never_t> s5 =
    asio::require(execution::stage(test_ex(ctx, 3)),
        execution::blocking.never);
  ASIO_CHECK(test_log::copies == 0);
  ASIO_CHECK(s5.get_inner_executor() == test_ex(ctx, 3));

  ASIO_CHECK(test_log::order_len == 0);

  execution::staging_executor<const test_ex&> s6(ex);
  ASIO_CHECK(&s6.get_inner_executor() == &ex);
  test_log::reset();
  execution::staging_executor<test_ex> s7(ex);
  ASIO_CHECK(test_log::copies == 1);

#if defined(__cpp_deduction_guides) && (__cpp_deduction_guides >= 201606)
  execution::staging_executor s8(ex);
  ASIO_CHECK((std::is_same<decltype(s8),
        execution::staging_executor<const test_ex&>>::value));
  execution::staging_executor s9(test_ex(ctx, 4));
  ASIO_CHECK((std::is_same<decltype(s9),
        execution::staging_executor<test_ex>>::value));
#endif // defined(__cpp_deduction_guides) && (__cpp_deduction_guides >= 201606)
}

void staging_executor_recording_test()
{
  io_context ctx;
  test_ex ex(ctx, 1);
  test_allocator<void> alloc(7);

  test_log::reset();
  typedef execution::staging_executor<const test_ex&,
    execution::allocator_t<test_allocator<void>>, continuation_t, never_t>
      staged_type;
  staged_type s = asio::require(
      asio::require(
        asio::require(execution::stage(ex), execution::allocator(alloc)),
        execution::relationship.continuation),
      execution::blocking.never);
  ASIO_CHECK(test_log::order_len == 0);

  test_executor<never_t, continuation_t> c = s.commit();
  ASIO_CHECK(std::strcmp(test_log::order, "arb") == 0);
  ASIO_CHECK((c == test_executor<never_t, continuation_t>(ctx, 1)));

  test_log::reset();
  execution::staging_executor<const test_ex&, possibly_t> s2 =
    asio::require(
        asio::require(execution::stage(ex), execution::blocking.never),
        execution::blocking.possibly);
  test_ex c2 = s2.commit();
  ASIO_CHECK(test_log::order_len == 0);
  ASIO_CHECK(c2 == ex);
  ASIO_CHECK(asio::query(s2, execution::blocking)
      == execution::blocking.possibly);

  execution::staging_executor<test_executor<possibly_t, continuation_t>,
    fork_t> s3 =
      asio::prefer(
          asio::require(execution::stage(test_executor<possibly_t,
              continuation_t>(ctx)), execution::relationship.continuation),
          execution::relationship.fork);
  ASIO_CHECK(asio::query(s3, execution::relationship)
      == execution::relationship.fork);

  execution::staging_executor<const test_ex&, fork_t> s4 =
    asio::require(execution::stage(ex), execution::relationship.fork);
  test_ex c4 = s4.commit();
  ASIO_CHECK(c4 == ex);
}

void staging_executor_sequence_accrual_test()
{
  io_context ctx;
  test_ex ex(ctx, 1);

  test_log::reset();
  execution::staging_executor<const test_ex&, never_t, continuation_t> s1 =
    asio::require(
        asio::require(execution::stage(ex), execution::blocking.never),
        execution::relationship.continuation);
  test_executor<never_t, continuation_t> c1 = s1.commit();
  ASIO_CHECK(std::strcmp(test_log::order, "br") == 0);
  ASIO_CHECK((c1 == test_executor<never_t, continuation_t>(ctx, 1)));

  test_log::reset();
  execution::staging_executor<const test_ex&, continuation_t, never_t> s2 =
    asio::require(
        asio::require(execution::stage(ex),
          execution::relationship.continuation),
        execution::blocking.never);
  test_executor<never_t, continuation_t> c2 = s2.commit();
  ASIO_CHECK(std::strcmp(test_log::order, "rb") == 0);
  ASIO_CHECK((c2 == test_executor<never_t, continuation_t>(ctx, 1)));

  test_log::reset();
  execution::staging_executor<const test_ex&, never_t, continuation_t> s3 =
    execution::stage(ex).require(execution::blocking.possibly)
      .require(execution::relationship.continuation)
      .require(execution::blocking.never);
  ASIO_CHECK(asio::query(s3, execution::blocking)
      == execution::blocking.never);
  ASIO_CHECK(asio::query(s3, execution::relationship)
      == execution::relationship.continuation);
  test_executor<never_t, continuation_t> c3 = s3.commit();
  ASIO_CHECK(std::strcmp(test_log::order, "br") == 0);
  ASIO_CHECK((c3 == test_executor<never_t, continuation_t>(ctx, 1)));
}

void staging_executor_repeated_property_test()
{
  io_context ctx;
  test_ex ex(ctx, 1);

  test_log::reset();
  execution::staging_executor<const test_ex&, never_t> r1 =
    execution::stage(ex).require(execution::blocking.never)
      .require(execution::blocking.never);
  ASIO_CHECK(asio::query(r1, execution::blocking)
      == execution::blocking.never);

  test_executor<never_t, fork_t> c1 = r1.commit();
  ASIO_CHECK(std::strcmp(test_log::order, "b") == 0);
  ASIO_CHECK((c1 == test_executor<never_t, fork_t>(ctx, 1)));

  any_io_executor erased = ex;
  execution::staging_executor<const any_io_executor&,
    execution::prefer_only<fork_t>> r2 =
      execution::stage(erased).prefer(execution::relationship.fork)
        .prefer(execution::relationship.fork);
  ASIO_CHECK(asio::query(r2, execution::relationship)
      == execution::relationship.fork);

  execution::staging_executor<const test_ex&,
    execution::allocator_t<test_allocator<void>>> r3 =
      asio::require(
          asio::require(execution::stage(ex),
            execution::allocator(test_allocator<void>(1))),
          execution::allocator(test_allocator<void>(2)));
  ASIO_CHECK(asio::query(r3, execution::allocator)
      == test_allocator<void>(2));
}

void staging_executor_preference_test()
{
  io_context ctx;
  test_ex ex(ctx, 1);

  any_io_executor erased = ex;
  execution::staging_executor<const any_io_executor&,
    execution::prefer_only<fork_t>> p1 =
      asio::prefer(execution::stage(erased), execution::relationship.fork);
  ASIO_CHECK(asio::query(p1, execution::relationship)
      == execution::relationship.fork);

  execution::staging_executor<const test_ex&, fork_t> p2 =
    asio::prefer(
        asio::require(execution::stage(ex),
          execution::relationship.continuation),
        execution::relationship.fork);
  (void)p2;

  test_ex c2b = asio::require(execution::stage(ex),
      execution::relationship.continuation).prefer(
        execution::relationship.fork).commit();
  ASIO_CHECK(c2b == ex);

  bare_executor bare;
  execution::staging_executor<const bare_executor&> p3 =
    asio::prefer(execution::stage(bare),
        execution::relationship.continuation);
  ASIO_CHECK(&p3.get_inner_executor() == &bare);

  execution::staging_executor<const bare_executor&> p4 =
    asio::prefer(execution::stage(bare),
        execution::allocator(std::allocator<void>()));
  ASIO_CHECK(p4.get_inner_executor() == bare);
  ASIO_CHECK(!(p4.get_inner_executor() != bare));

  execution::staging_executor<const any_io_executor&, never_t> p5 =
    asio::prefer(
        asio::require(execution::stage(erased), execution::blocking.never),
        execution::allocator(std::allocator<void>()));
  (void)p5;

  ASIO_CHECK((!can_require<execution::staging_executor<const bare_executor&>,
      never_t>::value));
  ASIO_CHECK((!can_require<execution::staging_executor<const any_io_executor&>,
      fork_t>::value));
  ASIO_CHECK((can_prefer<execution::staging_executor<const any_io_executor&>,
      fork_t>::value));
}

void staging_executor_other_property_test()
{
  io_context ctx;
  io_context::executor_type typed = ctx.get_executor();
  any_io_executor erased = typed;

  ASIO_CHECK((std::is_same<
        decltype(asio::require(
            asio::require(execution::stage(typed),
              execution::blocking.never),
            execution::outstanding_work.tracked)),
        execution::staging_executor<const io_context::executor_type&,
          never_t, tracked_t>
      >::value));

  ASIO_CHECK((asio::require(
          asio::require(execution::stage(typed),
            execution::blocking.never),
          execution::outstanding_work.tracked).commit()
        == asio::require(
          asio::require(typed, execution::blocking.never),
          execution::outstanding_work.tracked)));

  {
    any_io_executor tracked = asio::prefer(
        asio::require(execution::stage(erased), execution::blocking.never),
        execution::outstanding_work.tracked);
    ASIO_CHECK(tracked == asio::prefer(
          asio::require(erased, execution::blocking.never),
          execution::outstanding_work.tracked));
    ASIO_CHECK(asio::query(tracked, execution::outstanding_work)
        == execution::outstanding_work.tracked);
  }

  bare_executor bare;
  ASIO_CHECK((std::is_same<
        decltype(asio::prefer(execution::stage(bare),
            execution::outstanding_work.tracked)),
        execution::staging_executor<const bare_executor&>>::value));
}

void staging_executor_query_test()
{
  io_context ctx;
  test_executor<possibly_t, continuation_t> ex(ctx, 1);
  test_allocator<void> alloc(3);

  execution::staging_executor<
    const test_executor<possibly_t, continuation_t>&> s0 =
      execution::stage(ex);

  ASIO_CHECK(asio::query(s0, execution::blocking)
      == execution::blocking.possibly);
  ASIO_CHECK(asio::query(s0, execution::relationship)
      == execution::relationship.continuation);
  ASIO_CHECK(&asio::query(s0, execution::context) == &ctx);

  typedef execution::staging_executor<
    const test_executor<possibly_t, continuation_t>&,
    never_t, fork_t,
    execution::allocator_t<test_allocator<void>>> s1_type;
  s1_type s1 = asio::prefer(
      asio::require(s0, execution::blocking.never),
      execution::relationship.fork, execution::allocator(alloc));
  ASIO_CHECK(asio::query(s1, execution::blocking)
      == execution::blocking.never);
  ASIO_CHECK(asio::query(s1, execution::relationship)
      == execution::relationship.fork);
  ASIO_CHECK(asio::query(s1, execution::allocator) == alloc);
  ASIO_CHECK(&asio::query(s1, execution::context) == &ctx);

  ASIO_CHECK((std::is_same<
        decay_t<decltype(asio::require(s1, execution::blocking.never))>,
        s1_type>::value));
  ASIO_CHECK(asio::require(s1, execution::blocking.never) == s1);

  any_io_executor erased = ex;
  ASIO_CHECK(asio::query(
        asio::require(execution::stage(erased), execution::blocking.never),
        execution::blocking) == execution::blocking.never);
  ASIO_CHECK(&asio::query(execution::stage(erased),
        execution::context) == &ctx);
}

void staging_executor_equality_test()
{
  io_context ctx;
  test_ex ex1(ctx, 1);
  test_ex ex1b(ctx, 1);
  test_ex ex2(ctx, 2);

  ASIO_CHECK(execution::stage(ex1) == execution::stage(ex1));
  ASIO_CHECK(execution::stage(ex1) == execution::stage(ex1b));
  ASIO_CHECK(execution::stage(ex1) != execution::stage(ex2));
  ASIO_CHECK(execution::stage(test_ex(ctx, 1))
      == execution::stage(test_ex(ctx, 1)));

  ASIO_CHECK(
      asio::require(execution::stage(ex1),
        execution::allocator(test_allocator<void>(1)))
      == asio::require(execution::stage(ex1),
        execution::allocator(test_allocator<void>(1))));
  ASIO_CHECK(
      asio::require(execution::stage(ex1),
        execution::allocator(test_allocator<void>(1)))
      != asio::require(execution::stage(ex1),
        execution::allocator(test_allocator<void>(2))));

  ASIO_CHECK(noexcept(execution::stage(ex1) == execution::stage(ex2)));
}

void staging_executor_commit_test()
{
  io_context ctx;
  test_ex ex(ctx, 1);

  test_log::reset();
  execution::staging_executor<const test_ex&> s0 = execution::stage(ex);
  test_ex c0 = s0.commit();
  ASIO_CHECK(test_log::copies == 1);
  ASIO_CHECK(c0 == ex);

  test_log::reset();
  test_ex c1 = execution::stage(test_ex(ctx, 2)).commit();
  ASIO_CHECK(test_log::copies == 0);
  ASIO_CHECK(c1 == test_ex(ctx, 2));

  any_io_executor erased = asio::require(
      execution::stage(ex), execution::blocking.never);
  ASIO_CHECK(erased.target_type()
      == typeid(test_executor<never_t, fork_t>));
  ASIO_CHECK(erased == any_io_executor(
        asio::require(ex, execution::blocking.never)));

  any_io_executor erased2(asio::require(
        execution::stage(ex), execution::blocking.never));
  ASIO_CHECK(erased2 == erased);

  any_io_executor erased3 = asio::require(
      execution::stage(erased2), execution::blocking.never);
  ASIO_CHECK(erased3 == erased);

  execution::any_executor<> any0 = asio::require(
      execution::stage(ex), execution::blocking.never);
  ASIO_CHECK(any0.target_type()
      == typeid(test_executor<never_t, fork_t>));

  test_executor<never_t, fork_t> typed = asio::require(
      execution::stage(ex), execution::blocking.never);
  ASIO_CHECK((typed == test_executor<never_t, fork_t>(ctx, 1)));
}

void staging_executor_execute_test()
{
  io_context ctx;
  int count = 0;

  test_ex ex(ctx, 1);
  asio::prefer(
      asio::require(execution::stage(ex), execution::blocking.never),
      execution::relationship.continuation,
      execution::allocator(std::allocator<void>())
    ).execute(increment_function{&count});
  execution::stage(ex).execute(increment_function{&count});

  any_io_executor erased = ex;
  asio::require(execution::stage(erased), execution::blocking.never)
    .execute(increment_function{&count});
  asio::prefer(
      asio::require(execution::stage(erased), execution::blocking.never),
      execution::relationship.fork,
      execution::allocator(std::allocator<void>())
    ).execute(increment_function{&count});
  asio::prefer(
      asio::require(execution::stage(erased), execution::blocking.never),
      execution::relationship.continuation,
      execution::allocator(std::allocator<void>())
    ).execute(increment_function{&count});
  asio::prefer(execution::stage(erased),
      execution::allocator(std::allocator<void>())
    ).execute(increment_function{&count});
  asio::prefer(execution::stage(erased), execution::blocking.possibly)
    .execute(increment_function{&count});

  any_io_executor erased_ctx = ctx.get_executor();
  asio::require(execution::stage(erased_ctx), execution::blocking.never)
    .execute(increment_function{&count});

  strand<any_io_executor> s(erased_ctx);
  any_io_executor erased_strand = s;
  asio::prefer(
      asio::require(execution::stage(erased_strand),
        execution::blocking.never),
      execution::relationship.fork
    ).execute(increment_function{&count});

  ctx.run();
  ASIO_CHECK(count == 9);

  thread_pool pool(1);
  any_io_executor erased_pool = pool.get_executor();
  int pool_count = 0;
  asio::prefer(
      asio::require(execution::stage(erased_pool),
        execution::blocking.never),
      execution::relationship.fork
    ).execute(increment_function{&pool_count});
  pool.wait();
  ASIO_CHECK(pool_count == 1);

  any_io_executor erased_test = test_ex(ctx, 5);
  test_log::reset();
  asio::prefer(
      asio::require(execution::stage(erased_test),
        execution::blocking.never),
      execution::relationship.continuation,
      execution::allocator(std::allocator<void>())
    ).execute(increment_function{&count});
  ASIO_CHECK(test_log::copies == 0);
  ASIO_CHECK(test_log::moves == 0);
  ASIO_CHECK(test_log::order[0] == 'b');

  test_log::reset();
  asio::require(execution::stage(erased_test), execution::blocking.never)
    .execute(increment_function{&count});
  ASIO_CHECK(test_log::copies == 0);
  ASIO_CHECK(test_log::moves == 0);
  ASIO_CHECK(std::strcmp(test_log::order, "b") == 0);

  test_log::reset();
  asio::prefer(execution::stage(erased_test), execution::blocking.possibly)
    .execute(increment_function{&count});
  ASIO_CHECK(test_log::copies + test_log::moves > 0);

  ctx.restart();
  ctx.run();
  ASIO_CHECK(count == 12);

  any_io_executor empty;
  bool caught = false;
  try
  {
    asio::require(execution::stage(empty), execution::blocking.never)
      .execute(increment_function{&count});
  }
  catch (const execution::bad_executor&)
  {
    caught = true;
  }
  ASIO_CHECK(caught);
}

struct recording_operation : asio::detail::operation
{
  int* count_;
  bool* wrapped_;

  recording_operation(int* count, bool* wrapped)
    : asio::detail::operation(&recording_operation::do_complete),
      count_(count),
      wrapped_(wrapped)
  {
  }

  static void do_complete(void* owner, asio::detail::operation* base,
      int /*error_value*/, std::size_t /*bytes_transferred*/)
  {
    recording_operation* o = static_cast<recording_operation*>(base);
    if (owner)
    {
      ++*o->count_;
      if (owner == static_cast<void*>(o))
        *o->wrapped_ = true;
    }
    delete o;
  }
};

void staging_executor_execute_ready_made_test()
{
  io_context ctx(1);

  int unstaged_count = 0;
  bool unstaged_wrapped = false;
  asio::prefer(
      asio::require(execution::stage(ctx.get_executor()),
        execution::blocking.never),
      execution::relationship.fork
    ).execute(asio::detail::executor_function(
        new recording_operation(&unstaged_count, &unstaged_wrapped)));

  any_io_executor erased = ctx.get_executor();
  int staged_count = 0;
  bool staged_wrapped = false;
  asio::prefer(
      asio::require(execution::stage(erased), execution::blocking.never),
      execution::relationship.fork
    ).execute(asio::detail::executor_function(
        new recording_operation(&staged_count, &staged_wrapped)));

  ctx.run();

  ASIO_CHECK(unstaged_count == 1);
  ASIO_CHECK(!unstaged_wrapped);
  ASIO_CHECK(staged_count == 1);
  ASIO_CHECK(!staged_wrapped);
}

void staging_executor_staged_order_test()
{
  io_context ctx(1);
  any_io_executor erased = ctx.get_executor();

  typedef decay_t<
      decltype(
        asio::prefer(
          asio::require(execution::stage(erased), execution::blocking.never),
          execution::relationship.fork))
    > never_first_type;

  typedef decay_t<
      decltype(
        asio::require(
          asio::prefer(execution::stage(erased),
            execution::relationship.fork),
          execution::blocking.never))
    > fork_first_type;

  static_assert(std::is_same<never_first_type,
      execution::staging_executor<const any_io_executor&, never_t,
        execution::prefer_only<fork_t>>>::value, "staging_executor");

  static_assert(std::is_same<fork_first_type,
      execution::staging_executor<const any_io_executor&,
        execution::prefer_only<fork_t>, never_t>>::value, "staging_executor");

  int never_first = 0;
  int fork_first = 0;
  bool never_first_wrapped = false;
  bool fork_first_wrapped = false;

  asio::prefer(
      asio::require(execution::stage(erased), execution::blocking.never),
      execution::relationship.fork
    ).execute(asio::detail::executor_function(
        new recording_operation(&never_first, &never_first_wrapped)));

  asio::require(
      asio::prefer(execution::stage(erased), execution::relationship.fork),
      execution::blocking.never
    ).execute(asio::detail::executor_function(
        new recording_operation(&fork_first, &fork_first_wrapped)));

  ctx.run();

  ASIO_CHECK(never_first == 1);
  ASIO_CHECK(!never_first_wrapped);
  ASIO_CHECK(fork_first == 1);
  ASIO_CHECK(!fork_first_wrapped);
}

template <typename Inner>
struct thin_executor
{
  Inner inner_;

  template <typename Function>
  void execute(Function&& f) const
  {
    inner_.execute(static_cast<Function&&>(f));
  }

  void execute(asio::detail::executor_function&& f) const
  {
    inner_.execute(static_cast<asio::detail::executor_function&&>(f));
  }

  template <typename Property>
  auto require(const Property& p) const
    -> thin_executor<
        decay_t<
          decltype(asio::require(std::declval<const Inner&>(), p))
        >
      >
  {
    return thin_executor<
        decay_t<
          decltype(asio::require(std::declval<const Inner&>(), p))
        >
      >{asio::require(inner_, p)};
  }

  template <typename Property>
  auto prefer(const Property& p) const
    -> thin_executor<
        decay_t<
          decltype(asio::prefer(std::declval<const Inner&>(), p))
        >
      >
  {
    return thin_executor<
        decay_t<
          decltype(asio::prefer(std::declval<const Inner&>(), p))
        >
      >{asio::prefer(inner_, p)};
  }

  template <typename Property>
  auto query(const Property& p) const
    -> decltype(asio::query(std::declval<const Inner&>(), p))
  {
    return asio::query(inner_, p);
  }

  friend bool operator==(const thin_executor& a,
      const thin_executor& b) noexcept
  {
    return a.inner_ == b.inner_;
  }

  friend bool operator!=(const thin_executor& a,
      const thin_executor& b) noexcept
  {
    return !(a.inner_ == b.inner_);
  }
};

namespace asio {
namespace traits {

#if !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

template <typename Inner>
struct equality_comparable<thin_executor<Inner>>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
};

#endif // !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

template <typename Inner, typename Function>
struct execute_member<thin_executor<Inner>, Function,
    void_t<
      decltype(std::declval<const Inner&>().execute(
        std::declval<Function&&>()))
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

template <typename Inner, typename Property>
struct query_member<thin_executor<Inner>, Property,
    void_t<
      decltype(asio::query(std::declval<const Inner&>(),
        std::declval<const Property&>()))
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept =
    is_nothrow_query<const Inner&, Property>::value;
  typedef decltype(asio::query(std::declval<const Inner&>(),
      std::declval<const Property&>())) result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)

template <typename Inner, typename Property>
struct require_member<thin_executor<Inner>, Property,
    void_t<
      decltype(asio::require(std::declval<const Inner&>(),
        std::declval<const Property&>()))
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef thin_executor<
      decay_t<
        decltype(asio::require(std::declval<const Inner&>(),
          std::declval<const Property&>()))
      >
    > result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_PREFER_MEMBER_TRAIT)

template <typename Inner, typename Property>
struct prefer_member<thin_executor<Inner>, Property,
    void_t<
      decltype(asio::prefer(std::declval<const Inner&>(),
        std::declval<const Property&>()))
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef thin_executor<
      decay_t<
        decltype(asio::prefer(std::declval<const Inner&>(),
          std::declval<const Property&>()))
      >
    > result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_PREFER_MEMBER_TRAIT)

} // namespace traits
} // namespace asio

void staging_executor_erased_target_ready_made_test()
{
  io_context ctx(1);

  any_io_executor erased = ctx.get_executor();
  typedef decay_t<
      decltype(
        asio::prefer(
          asio::require(execution::stage(erased), execution::blocking.never),
          execution::relationship.fork))
    > staged_type;
  typedef thin_executor<staged_type> thin_type;
  any_io_executor a(
      thin_type{
        asio::prefer(
          asio::require(execution::stage(erased), execution::blocking.never),
          execution::relationship.fork)});

  ASIO_CHECK(a.target_type() == typeid(thin_type));

  int count = 0;
  bool wrapped = false;
  a.execute(asio::detail::executor_function(
        new recording_operation(&count, &wrapped)));

  ctx.run();

  ASIO_CHECK(count == 1);
  ASIO_CHECK(!wrapped);
}

ASIO_TEST_SUITE
(
  "staging_executor",
  ASIO_TEST_CASE(staging_executor_ownership_test)
  ASIO_TEST_CASE(staging_executor_recording_test)
  ASIO_TEST_CASE(staging_executor_sequence_accrual_test)
  ASIO_TEST_CASE(staging_executor_repeated_property_test)
  ASIO_TEST_CASE(staging_executor_preference_test)
  ASIO_TEST_CASE(staging_executor_other_property_test)
  ASIO_TEST_CASE(staging_executor_query_test)
  ASIO_TEST_CASE(staging_executor_equality_test)
  ASIO_TEST_CASE(staging_executor_commit_test)
  ASIO_TEST_CASE(staging_executor_execute_test)
  ASIO_TEST_CASE(staging_executor_execute_ready_made_test)
  ASIO_TEST_CASE(staging_executor_staged_order_test)
  ASIO_TEST_CASE(staging_executor_erased_target_ready_made_test)
)
