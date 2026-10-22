//
// experimental/diagnostic_executor.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2026 Pinwhell <binarydetective@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained and zero-dependency.
#include "asio/experimental/diagnostic_executor.hpp"

#include "asio/io_context.hpp"
#include "asio/post.hpp"
#include "asio/strand.hpp"
#include "asio/execution/any_executor.hpp"
#include "../unit_test.hpp"
#include <vector>
#include <string>
#include <mutex>

struct test_policy
{
  static std::mutex& mutex()
  {
    static std::mutex m;
    return m;
  }

  static std::vector<std::string>& labels()
  {
    static std::vector<std::string> l;
    return l;
  }

  template <typename Label>
  static void on_submit(const Label& label) noexcept
  {
    std::lock_guard<std::mutex> lock(mutex());
    labels().push_back(std::string(label));
  }
};

void test_diagnostic_executor_basic()
{
  asio::io_context ioc;
  
  typedef asio::experimental::diagnostic_executor<
    asio::io_context::executor_type, const char*, test_policy> diag_ex_type;
    
  diag_ex_type ex(ioc.get_executor(), "test_label");

  ASIO_CHECK(ex.get_inner_executor() == ioc.get_executor());

  bool executed = false;
  ex.execute([&executed]() { executed = true; });

  ASIO_CHECK(test_policy::labels().size() == 1);
  ASIO_CHECK(test_policy::labels().back() == "test_label");

  ioc.run();
  ASIO_CHECK(executed);
}

void test_diagnostic_executor_traits()
{
  asio::io_context ioc;
  auto ex = asio::experimental::make_diagnostic_executor(ioc.get_executor(), "test");

  // Verify it is recognized as an executor (both old and new models).
  ASIO_CHECK(asio::is_executor<decltype(ex)>::value);
  ASIO_CHECK(asio::execution::is_executor<decltype(ex)>::value);

  // Verify property query propagation.
  ASIO_CHECK(&asio::query(ex, asio::execution::context) == &ioc);
  ASIO_CHECK(asio::query(ex, asio::execution::blocking) == asio::execution::blocking.possibly);

  // Verify requirement propagation (should return a new diagnostic_executor).
  auto ex2 = asio::require(ex, asio::execution::blocking.never);
  ASIO_CHECK(asio::query(ex2, asio::execution::blocking) == asio::execution::blocking.never);
  
  typedef asio::experimental::diagnostic_executor<
    asio::decay_t<asio::require_result_t<asio::io_context::executor_type, decltype(asio::execution::blocking.never)>>,
    const char*, asio::experimental::null_diagnostic_policy> expected_type;

  (void)ex2; // Silence unused variable warning.
  (void)sizeof(expected_type);
}

void test_diagnostic_executor_compatibility()
{
  asio::io_context ioc;
  auto ex = asio::experimental::make_diagnostic_executor(ioc.get_executor(), "compat");

  bool executed = false;
  asio::post(ex, [&executed]() { executed = true; });

  ioc.run();
  ASIO_CHECK(executed);
}

void test_diagnostic_executor_any_executor()
{
  asio::io_context ioc;
  asio::execution::any_executor<
      asio::execution::blocking_t::possibly_t,
      asio::execution::outstanding_work_t::tracked_t,
      asio::execution::relationship_t::fork_t
    > ex = ioc.get_executor();
    
  auto diag_ex = asio::experimental::make_diagnostic_executor(ex, "any_ex");

  ASIO_CHECK(asio::execution::is_executor<decltype(diag_ex)>::value);

  bool executed = false;
  diag_ex.execute([&executed]() { executed = true; });

  ioc.run();
  ASIO_CHECK(executed);
}

void test_diagnostic_executor_strand()
{
  asio::io_context ioc;
  auto s = asio::make_strand(ioc);
  auto ex = asio::experimental::make_diagnostic_executor(s, "strand");

  bool executed = false;
  ex.execute([&executed]() { executed = true; });

  ioc.run();
  ASIO_CHECK(executed);
}

template <typename T>
struct test_allocator
{
  typedef T value_type;
  test_allocator() noexcept {}
  template <typename U> test_allocator(const test_allocator<U>&) noexcept {}
  bool operator==(const test_allocator&) const { return true; }
  bool operator!=(const test_allocator&) const { return false; }
  T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T))); }
  void deallocate(T* p, std::size_t) { ::operator delete(p); }
};

void test_diagnostic_executor_allocator()
{
  asio::io_context ioc;
  auto ex = asio::experimental::make_diagnostic_executor(ioc.get_executor(), "alloc");
  test_allocator<void> alloc;

  bool executed = false;
#if !defined(ASIO_NO_TS_EXECUTORS)
  ex.post([&executed]() { executed = true; }, alloc);
#else // !defined(ASIO_NO_TS_EXECUTORS)
  asio::post(ex, [&executed]() { executed = true; });
#endif // !defined(ASIO_NO_TS_EXECUTORS)

  ioc.run();
  ASIO_CHECK(executed);
}

void test_diagnostic_executor_transparency()
{
  asio::io_context ioc;
  typedef asio::experimental::diagnostic_executor<
    asio::io_context::executor_type> diag_ex_type;

  ASIO_CHECK((asio::is_same<diag_ex_type::nested_executor_type, asio::io_context::executor_type>::value));
}

void test_diagnostic_executor_legacy_1arg()
{
  asio::io_context ioc;
  auto ex = asio::experimental::make_diagnostic_executor(ioc.get_executor(), "legacy_1arg");

  bool dispatch_executed = false;
  ex.dispatch([&]() { dispatch_executed = true; });

  bool post_executed = false;
  ex.post([&]() { post_executed = true; });

  bool defer_executed = false;
  ex.defer([&]() { defer_executed = true; });

  ioc.run();
  ASIO_CHECK(dispatch_executed);
  ASIO_CHECK(post_executed);
  ASIO_CHECK(defer_executed);
}

ASIO_TEST_SUITE
(
  "experimental/diagnostic_executor",
  ASIO_TEST_CASE(test_diagnostic_executor_basic)
  ASIO_TEST_CASE(test_diagnostic_executor_traits)
  ASIO_TEST_CASE(test_diagnostic_executor_compatibility)
  ASIO_TEST_CASE(test_diagnostic_executor_any_executor)
  ASIO_TEST_CASE(test_diagnostic_executor_strand)
  ASIO_TEST_CASE(test_diagnostic_executor_allocator)
  ASIO_TEST_CASE(test_diagnostic_executor_transparency)
  ASIO_TEST_CASE(test_diagnostic_executor_legacy_1arg)
)
