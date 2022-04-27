//
// experimental/deferred.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2022 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include <asio/steady_timer.hpp>
#include "asio/experimental/append.hpp"
#include "asio/experimental/parallel_group.hpp"
#include "asio/io_context.hpp"
#include "asio/post.hpp"
#include "../unit_test.hpp"

struct foobar
{
  template<typename ... Args>
  void operator()(Args &&... ) const;

};

struct test_wrap
{
  int cnt = 0;

  std::shared_ptr<asio::steady_timer> timer;

  test_wrap(int cnt, asio::any_io_executor exec, int ms )
    : cnt(cnt),
      timer(std::make_shared<asio::steady_timer>(exec, std::chrono::milliseconds(ms)))
  {
  }

  template<typename Token>
  auto operator()(Token tk) const
  {
    return timer->async_wait(asio::experimental::append(std::move(tk), cnt));
  }
};


void variadic_test()
{
  using namespace asio::experimental;

  asio::io_context ctx;

  std::vector<test_wrap> wrap1 {
          test_wrap{1, ctx.get_executor(), 100},
          test_wrap{2, ctx.get_executor(), 60},
          test_wrap{3, ctx.get_executor(), 80},
          test_wrap{4, ctx.get_executor(), 20},
          test_wrap{5, ctx.get_executor(), 40}
  };

  auto grp = make_parallel_group(wrap1);
  int called = 0;
  grp.async_wait(wait_for_all(), [&](std::vector<std::size_t> seq,
                                    std::vector<std::tuple<asio::error_code, int>> res)
  {
    called++;
    ASIO_CHECK_MESSAGE(seq.size() == res.size(), seq.size() << " == " << res.size());
    ASIO_CHECK(seq.size() == 5u);
    std::vector<std::size_t> cmp{3,4,1,2,0};
    ASIO_CHECK(seq == cmp);
    ASIO_CHECK(!std::get<0>(res.at(0)));
    ASIO_CHECK(!std::get<0>(res.at(1)));
    ASIO_CHECK(!std::get<0>(res.at(2)));
    ASIO_CHECK(!std::get<0>(res.at(3)));
    ASIO_CHECK(!std::get<0>(res.at(4)));

    ASIO_CHECK(std::get<1>(res.at(0)) = 1);
    ASIO_CHECK(std::get<1>(res.at(1)) = 2);
    ASIO_CHECK(std::get<1>(res.at(2)) = 3);
    ASIO_CHECK(std::get<1>(res.at(3)) = 4);
    ASIO_CHECK(std::get<1>(res.at(4)) = 5);
  });


  std::vector<test_wrap> wrap2 {
          test_wrap{1, ctx.get_executor(), 60},
          test_wrap{2, ctx.get_executor(), 20},
          test_wrap{3, ctx.get_executor(), 10},
          test_wrap{4, ctx.get_executor(), 80},

  };
  auto grp2 = make_parallel_group(wrap2);
  grp2.async_wait(wait_for_one(), [&](std::vector<std::size_t> seq,
                                     std::vector<std::tuple<asio::error_code, int>> res)
  {
    called++;
    ASIO_CHECK_MESSAGE(seq.size() == res.size(), seq.size() << " == " << res.size());
    ASIO_CHECK_MESSAGE(seq.size() == 4u, seq.size());
    ASIO_CHECK(seq[0] == 2u);
    ASIO_CHECK(std::get<0>(res.at(0)) == asio::error::operation_aborted);
    ASIO_CHECK(std::get<0>(res.at(1)) == asio::error::operation_aborted);
    ASIO_CHECK(!std::get<0>(res.at(2)));
    ASIO_CHECK(std::get<0>(res.at(3)) == asio::error::operation_aborted);

    ASIO_CHECK(std::get<1>(res.at(0)) = 2);

  });

  ctx.run();
  ASIO_CHECK(called == 2);
}

ASIO_TEST_SUITE
(
  "experimental/parallel_group",
  ASIO_TEST_CASE(variadic_test)
)
