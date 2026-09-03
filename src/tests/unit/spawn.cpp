//
// spawn.cpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2024 Casey Bodley (cbodley at redhat dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/spawn.hpp"

#include "unit_test.hpp"

#if defined(ASIO_HAS_BOOST_CONTEXT_FIBER) \
  || defined(ASIO_HAS_ASM_FIBER)

#include <memory>
#include <stdexcept>
#include <vector>
#include "archetypes/async_ops.hpp"
#include "asio/any_completion_handler.hpp"
#include "asio/bind_cancellation_slot.hpp"
#include "asio/deferred.hpp"
#include "asio/dispatch.hpp"
#include "asio/detached.hpp"
#include "asio/io_context.hpp"
#include "asio/post.hpp"
#include "asio/steady_timer.hpp"

#if defined(ASIO_HAS_ASM_FIBER)
# include "asio/detail/default_stack_allocator.hpp"
#endif // defined(ASIO_HAS_ASM_FIBER)

#if defined(ASIO_HAS_FIBER_GUARD_PAGE)
# include <cerrno>
# include <unistd.h>
#endif // defined(ASIO_HAS_FIBER_GUARD_PAGE)

void void_returning_coroutine(asio::yield_context)
{
}

int int_returning_coroutine(asio::yield_context)
{
  return 42;
}

void test_spawn_with_any_completion_handler()
{
  asio::io_context ctx;

  bool called = false;
  asio::spawn(ctx, void_returning_coroutine,
      asio::any_completion_handler<void(std::exception_ptr)>(
        [&](std::exception_ptr)
        {
          called = true;
        }));

  ASIO_CHECK(!called);

  ctx.run();

  ASIO_CHECK(called);

  int result = 0;
  asio::spawn(ctx, int_returning_coroutine,
      asio::any_completion_handler<void(std::exception_ptr, int)>(
        [&](std::exception_ptr, int i)
        {
          result = i;
        }));

  ASIO_CHECK(result == 0);

  ctx.restart();
  ctx.run();

  ASIO_CHECK(result == 42);
}

void test_spawn_deferred()
{
  asio::io_context ctx;

  {
    bool called = false;
    auto fn = asio::spawn(ctx, void_returning_coroutine, asio::deferred);

    fn([&](std::exception_ptr)
        {
          called = true;
        });

    ASIO_CHECK(!called);

    ctx.poll();

    ASIO_CHECK(ctx.stopped());
    ASIO_CHECK(called);
  }
  {
    int result = 0;
    auto fn = asio::spawn(ctx, int_returning_coroutine, asio::deferred);

    fn([&](std::exception_ptr, int i)
        {
          result = i;
        });

    ASIO_CHECK(result == 0);

    ctx.restart();
    ctx.poll();

    ASIO_CHECK(ctx.stopped());
    ASIO_CHECK(result == 42);
  }
}

void sleeping_coroutine(asio::yield_context yield)
{
  asio::steady_timer timer(yield.get_executor(),
      asio::steady_timer::time_point::max());
  timer.async_wait(yield);
}

void test_spawn_cancel()
{
#if !defined(ASIO_NO_EXCEPTIONS)
  asio::cancellation_signal sig;
  asio::io_context ctx;

  std::exception_ptr result = nullptr;
  bool called = false;
  asio::spawn(ctx, sleeping_coroutine,
      asio::bind_cancellation_slot(sig.slot(),
        [&](std::exception_ptr e)
        {
          result = e;
          called = true;
        }));

  ctx.poll();
  ASIO_CHECK(!ctx.stopped());

  ASIO_CHECK(!called);
  ASIO_CHECK(result == nullptr);

  sig.emit(asio::cancellation_type::all);

  ctx.poll();
  ASIO_CHECK(ctx.stopped());

  ASIO_CHECK(called);
  ASIO_CHECK(result != nullptr);
  try
  {
    std::rethrow_exception(result);
  }
  catch (const std::system_error& e)
  {
    ASIO_CHECK(e.code() == asio::error::operation_aborted);
  }
  catch (...)
  {
    ASIO_ERROR("expected system_error");
  }
#endif // !defined(ASIO_NO_EXCEPTIONS)
}

#if !defined(ASIO_NO_EXCEPTIONS)

void throwing_coroutine(asio::yield_context)
{
  throw std::runtime_error("oops");
}

#endif // !defined(ASIO_NO_EXCEPTIONS)

void test_spawn_exception()
{
#if !defined(ASIO_NO_EXCEPTIONS)
  asio::io_context ctx;

  std::exception_ptr result = nullptr;
  bool called = false;
  asio::spawn(ctx, throwing_coroutine,
      [&](std::exception_ptr e)
      {
        result = e;
        called = true;
      });

  ctx.poll();
  ASIO_CHECK(ctx.stopped());

  ASIO_CHECK(called);
  ASIO_CHECK(result != nullptr);
  try
  {
    std::rethrow_exception(result);
  }
  catch (const std::runtime_error&)
  {
    // ok
  }
  catch (...)
  {
    ASIO_ERROR("expected runtime_error");
  }
#endif // !defined(ASIO_NO_EXCEPTIONS)
}

std::unique_ptr<int> factory_coroutine(asio::yield_context)
{
  return std::unique_ptr<int>(new int(42));
}

void test_spawn_return_move_only()
{
  asio::io_context ctx;

  std::unique_ptr<int> result;
  bool called = false;
  asio::spawn(ctx, factory_coroutine,
      [&](std::exception_ptr, std::unique_ptr<int> r)
      {
        result = std::move(r);
        called = true;
      });

  ctx.poll();
  ASIO_CHECK(ctx.stopped());

  ASIO_CHECK(called);
  ASIO_CHECK(result);
  ASIO_CHECK(*result == 42);
}

#if !defined(ASIO_NO_EXCEPTIONS)

int coroutine_using_async_ops_0(asio::yield_context yield)
{
  using namespace archetypes;

  try
  {
    async_op_0(yield);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    async_op_ec_0(true, yield);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    async_op_ec_0(false, yield);
    ASIO_CHECK(false);
  }
  catch (asio::system_error& e)
  {
    ASIO_CHECK(e.code() == asio::error::operation_aborted);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    async_op_ex_0(true, yield);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    async_op_ex_0(false, yield);
    ASIO_CHECK(false);
  }
  catch (std::exception& e)
  {
    ASIO_CHECK(e.what() == std::string("blah"));
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  return 123;
}

int coroutine_using_async_ops_1(asio::yield_context yield)
{
  using namespace archetypes;

  try
  {
    int i = async_op_1(yield);
    ASIO_CHECK(i == 42);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    int i = async_op_ec_1(true, yield);
    ASIO_CHECK(i == 42);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    int i = async_op_ec_1(false, yield);
    ASIO_CHECK(false);
    (void)i;
  }
  catch (asio::system_error& e)
  {
    ASIO_CHECK(e.code() == asio::error::operation_aborted);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    int i = async_op_ex_1(true, yield);
    ASIO_CHECK(i == 42);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    int i = async_op_ex_1(false, yield);
    ASIO_CHECK(false);
    (void)i;
  }
  catch (std::exception& e)
  {
    ASIO_CHECK(e.what() == std::string("blah"));
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  return 234;
}

int coroutine_using_async_ops_2(asio::yield_context yield)
{
  using namespace archetypes;

  try
  {
    int i;
    double d;
    std::tie(i, d) = async_op_2(yield);
    ASIO_CHECK(i == 42);
    ASIO_CHECK(d == 2.0);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    int i;
    double d;
    std::tie(i, d) = async_op_ec_2(true, yield);
    ASIO_CHECK(i == 42);
    ASIO_CHECK(d == 2.0);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    std::tuple<int, double> t = async_op_ec_2(false, yield);
    ASIO_CHECK(false);
    (void)t;
  }
  catch (asio::system_error& e)
  {
    ASIO_CHECK(e.code() == asio::error::operation_aborted);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    int i;
    double d;
    std::tie(i, d) = async_op_ex_2(true, yield);
    ASIO_CHECK(i == 42);
    ASIO_CHECK(d == 2.0);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    std::tuple<int, double> t = async_op_ex_2(false, yield);
    ASIO_CHECK(false);
    (void)t;
  }
  catch (std::exception& e)
  {
    ASIO_CHECK(e.what() == std::string("blah"));
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  return 345;
}

int coroutine_using_async_ops_3(asio::yield_context yield)
{
  using namespace archetypes;

  try
  {
    int i;
    double d;
    char c;
    std::tie(i, d, c) = async_op_3(yield);
    ASIO_CHECK(i == 42);
    ASIO_CHECK(d == 2.0);
    ASIO_CHECK(c == 'a');
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    int i;
    double d;
    char c;
    std::tie(i, d, c) = async_op_ec_3(true, yield);
    ASIO_CHECK(i == 42);
    ASIO_CHECK(d == 2.0);
    ASIO_CHECK(c == 'a');
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    std::tuple<int, double, char> t = async_op_ec_3(false, yield);
    ASIO_CHECK(false);
    (void)t;
  }
  catch (asio::system_error& e)
  {
    ASIO_CHECK(e.code() == asio::error::operation_aborted);
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    int i;
    double d;
    char c;
    std::tie(i, d, c) = async_op_ex_3(true, yield);
    ASIO_CHECK(i == 42);
    ASIO_CHECK(d == 2.0);
    ASIO_CHECK(c == 'a');
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  try
  {
    std::tuple<int, double, char> t = async_op_ex_3(false, yield);
    ASIO_CHECK(false);
    (void)t;
  }
  catch (std::exception& e)
  {
    ASIO_CHECK(e.what() == std::string("blah"));
  }
  catch (...)
  {
    ASIO_CHECK(false);
  }

  return 456;
}

#endif // !defined(ASIO_NO_EXCEPTIONS)

void test_spawn_async_ops()
{
#if !defined(ASIO_NO_EXCEPTIONS)
  asio::io_context ctx;

  bool called = false;
  asio::spawn(ctx, coroutine_using_async_ops_0,
      [&](std::exception_ptr, int i)
      {
        ASIO_CHECK(i == 123);
        called = true;
      });

  ASIO_CHECK(!called);

  ctx.run();

  ASIO_CHECK(called);

  called = false;
  asio::spawn(ctx, coroutine_using_async_ops_1,
      [&](std::exception_ptr, int i)
      {
        ASIO_CHECK(i == 234);
        called = true;
      });

  ASIO_CHECK(!called);

  ctx.restart();
  ctx.run();

  ASIO_CHECK(called);

  called = false;
  asio::spawn(ctx, coroutine_using_async_ops_2,
      [&](std::exception_ptr, int i)
      {
        ASIO_CHECK(i == 345);
        called = true;
      });

  ASIO_CHECK(!called);

  ctx.restart();
  ctx.run();

  ASIO_CHECK(called);

  called = false;
  asio::spawn(ctx, coroutine_using_async_ops_3,
      [&](std::exception_ptr, int i)
      {
        ASIO_CHECK(i == 456);
        called = true;
      });

  ASIO_CHECK(!called);

  ctx.restart();
  ctx.run();

  ASIO_CHECK(called);
#endif // !defined(ASIO_NO_EXCEPTIONS)
}

int stack_object_count = 0;

struct stack_object
{
  stack_object() { ++stack_object_count; }
  stack_object(const stack_object&) { ++stack_object_count; }
  ~stack_object() { --stack_object_count; }
};

void test_spawn_io_context_destroyed()
{
  std::exception_ptr result = nullptr;
  bool called = false;

  ASIO_CHECK(stack_object_count == 0);

  {
    asio::io_context ctx;

    asio::spawn(ctx,
        [](asio::yield_context yield)
        {
          stack_object a;
          stack_object b;

          asio::steady_timer timer(yield.get_executor());
          timer.expires_at(asio::steady_timer::time_point::max());
          timer.async_wait(yield);

          ASIO_CHECK(false);
        },
        [&](std::exception_ptr e)
        {
          result = e;
          called = true;
        });

    ctx.poll();
    ASIO_CHECK(!ctx.stopped());

    ASIO_CHECK(stack_object_count == 2);
    ASIO_CHECK(!called);

  }

#if !defined(ASIO_NO_EXCEPTIONS)
  ASIO_CHECK(stack_object_count == 0);
#else // !defined(ASIO_NO_EXCEPTIONS)
  ASIO_CHECK(stack_object_count == 2);
  stack_object_count = 0;
#endif // !defined(ASIO_NO_EXCEPTIONS)

  ASIO_CHECK(!called);
  ASIO_CHECK(result == nullptr);
}

void test_spawn_io_context_destroyed_in_catch()
{
#if !defined(ASIO_NO_EXCEPTIONS)
  ASIO_CHECK(stack_object_count == 0);

  bool called = false;
  bool reached_catch = false;

  {
    asio::io_context ctx;

    asio::spawn(ctx,
        [&](asio::yield_context yield)
        {
          try
          {
            throw std::runtime_error("test");
          }
          catch (const std::runtime_error&)
          {
            reached_catch = true;
            stack_object a;

            asio::steady_timer timer(yield.get_executor());
            timer.expires_at(asio::steady_timer::time_point::max());
            timer.async_wait(yield);

            ASIO_CHECK(false);
          }
        },
        [&](std::exception_ptr) { called = true; });

    ctx.poll();
    ASIO_CHECK(reached_catch);
    ASIO_CHECK(stack_object_count == 1);
    ASIO_CHECK(!called);
  }

  ASIO_CHECK(stack_object_count == 0);
  ASIO_CHECK(!called);
#endif // !defined(ASIO_NO_EXCEPTIONS)
}

void test_spawn_nested()
{
  asio::io_context ctx;

  int order = 0;
  int outer_at = 0;
  int inner_at = 0;

  asio::spawn(ctx,
      [&](asio::yield_context yield)
      {
        outer_at = ++order;

        asio::spawn(yield,
            [&](asio::yield_context)
            {
              inner_at = ++order;
            }, yield);
      }, asio::detached);

  ctx.run();

  ASIO_CHECK(outer_at == 1);
  ASIO_CHECK(inner_at == 2);
}

void test_spawn_many()
{
  const int n = 16;
  const int rounds = 4;

  asio::io_context ctx;
  std::vector<int> counts(n, 0);

  for (int i = 0; i < n; ++i)
  {
    asio::spawn(ctx,
        [&counts, i](asio::yield_context yield)
        {
          for (int j = 0; j < rounds; ++j)
          {
            ++counts[i];
            asio::post(yield);
          }
        }, asio::detached);
  }

  ctx.run();

  for (int i = 0; i < n; ++i)
    ASIO_CHECK(counts[i] == rounds);
}

void test_spawn_completes_out_of_line()
{
  asio::io_context ctx;

  int depth = 0;
  int max_depth = 0;
  bool called = false;

  asio::post(ctx,
      [&]
      {
        ++depth;
        max_depth = depth > max_depth ? depth : max_depth;

        asio::spawn(ctx,
            [](asio::yield_context)
            {
            },
            [&](std::exception_ptr)
            {
              ++depth;
              max_depth = depth > max_depth ? depth : max_depth;
              called = true;
              --depth;
            });

        --depth;
      });

  ctx.run();

  ASIO_CHECK(called);
  ASIO_CHECK(max_depth == 1);
}

void test_spawn_inline_completions()
{
  asio::io_context ctx;

  int completed = 0;
  int destroyed = 0;

  struct tracker
  {
    int& destroyed_;
    ~tracker() { ++destroyed_; }
  };

  asio::spawn(ctx,
      [&](asio::yield_context yield)
      {
        tracker t{destroyed};
        for (int i = 0; i < 16; ++i)
          asio::dispatch(yield);
        completed = 1;
      }, asio::detached);

  ctx.run();

  ASIO_CHECK(completed == 1);
  ASIO_CHECK(destroyed == 1);
}

void test_spawn_inline_completions_abandoned()
{
#if !defined(ASIO_NO_EXCEPTIONS)
  int destroyed = 0;

  struct tracker
  {
    int& destroyed_;
    ~tracker() { ++destroyed_; }
  };

  {
    asio::io_context ctx;
    asio::steady_timer timer(ctx);

    asio::spawn(ctx,
        [&](asio::yield_context yield)
        {
          tracker t{destroyed};
          for (int i = 0; i < 8; ++i)
            asio::dispatch(yield);
          timer.expires_after(asio::chrono::hours(1));
          timer.async_wait(yield);
        }, asio::detached);

    ctx.run_one();
  }

  ASIO_CHECK(destroyed == 1);
#endif // !defined(ASIO_NO_EXCEPTIONS)
}

struct test_stack_allocator
{
  struct context
  {
    void* sp;
    std::size_t size;
    void* base;

    context(void* sp_arg, std::size_t size_arg, void* base_arg)
      : sp(sp_arg), size(size_arg), base(base_arg)
    {
    }

    context(const context&) = default;
    context() = delete;
    context& operator=(const context&) = delete;
  };

  int* allocated;
  int* deallocated;

  context allocate() const
  {
    const std::size_t size = ASIO_DEFAULT_FIBER_STACK_SIZE;
    char* base = new char[size];
    ++*allocated;
    return context(base + size, size, base);
  }

  void deallocate(context& c) const
  {
    ++*deallocated;
    delete[] static_cast<char*>(c.base);
  }
};

void test_spawn_stack_allocator()
{
  int allocated = 0;
  int deallocated = 0;
  int count = 0;

  {
    asio::io_context ctx;
    test_stack_allocator allocator = { &allocated, &deallocated };

    asio::spawn(ctx, asio::allocator_arg_t(), allocator,
        [&](asio::yield_context yield)
        {
          for (int i = 0; i < 4; ++i)
          {
            asio::post(yield);
            ++count;
          }
        }, asio::detached);

    ASIO_CHECK(allocated == 1);
    ASIO_CHECK(deallocated == 0);

    ctx.run();
  }

  ASIO_CHECK(count == 4);
  ASIO_CHECK(allocated == 1);
  ASIO_CHECK(deallocated == 1);
}

void test_default_stack_allocator()
{
#if defined(ASIO_HAS_ASM_FIBER)
  typedef asio::detail::default_stack_allocator allocator_type;
  allocator_type allocator(ASIO_DEFAULT_FIBER_STACK_SIZE);

  // Successive stacks start at different offsets, so try every one of them.
  for (int i = 0; i < allocator_type::colour_count; ++i)
  {
    allocator_type::stack_context stack = allocator.allocate();
    char* sp = static_cast<char*>(stack.sp);

    ASIO_CHECK(sp - stack.size == static_cast<char*>(stack.base));
    ASIO_CHECK(stack.size >= ASIO_DEFAULT_FIBER_STACK_SIZE);
    sp[-1] = 0;

    allocator.deallocate(stack);
  }
#endif // defined(ASIO_HAS_ASM_FIBER)
}

void test_stack_guard_page()
{
#if defined(ASIO_HAS_FIBER_GUARD_PAGE)
  asio::detail::default_stack_allocator allocator(
      ASIO_DEFAULT_FIBER_STACK_SIZE);
  asio::detail::default_stack_allocator::stack_context stack =
    allocator.allocate();

  int fds[2];
  ASIO_CHECK(::pipe(fds) == 0);

  errno = 0;
  ASIO_CHECK(::write(fds[1], stack.base, 1) == -1);
  ASIO_CHECK(errno == EFAULT);

  ASIO_CHECK(::write(fds[1],
        static_cast<char*>(stack.base) + allocator.page_size(), 1) == 1);

  ASIO_CHECK(::close(fds[0]) == 0);
  ASIO_CHECK(::close(fds[1]) == 0);
  allocator.deallocate(stack);
#endif // defined(ASIO_HAS_FIBER_GUARD_PAGE)
}

ASIO_TEST_SUITE
(
  "spawn",
  ASIO_TEST_CASE(test_spawn_with_any_completion_handler)
  ASIO_TEST_CASE(test_spawn_deferred)
  ASIO_TEST_CASE(test_spawn_cancel)
  ASIO_TEST_CASE(test_spawn_exception)
  ASIO_TEST_CASE(test_spawn_return_move_only)
  ASIO_TEST_CASE(test_spawn_async_ops)
  ASIO_TEST_CASE(test_spawn_completes_out_of_line)
  ASIO_TEST_CASE(test_spawn_io_context_destroyed)
  ASIO_TEST_CASE(test_spawn_io_context_destroyed_in_catch)
  ASIO_TEST_CASE(test_spawn_nested)
  ASIO_TEST_CASE(test_spawn_many)
  ASIO_TEST_CASE(test_spawn_inline_completions)
  ASIO_TEST_CASE(test_spawn_inline_completions_abandoned)
  ASIO_TEST_CASE(test_spawn_stack_allocator)
  ASIO_TEST_CASE(test_default_stack_allocator)
  ASIO_TEST_CASE(test_stack_guard_page)
)

#else // defined(ASIO_HAS_BOOST_CONTEXT_FIBER)

ASIO_TEST_SUITE
(
  "spawn",
  ASIO_TEST_CASE(null_test)
)

#endif // defined(ASIO_HAS_BOOST_CONTEXT_FIBER)
