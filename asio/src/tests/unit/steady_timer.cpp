//
// steady_timer.cpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Prevent link dependency on the Boost.System library.
#if !defined(BOOST_SYSTEM_NO_DEPRECATED)
#define BOOST_SYSTEM_NO_DEPRECATED
#endif // !defined(BOOST_SYSTEM_NO_DEPRECATED)

// Test that header file is self-contained.
#include "asio/error_code.hpp"
#include "asio/steady_timer.hpp"

#include "unit_test.hpp"

namespace asio
{
template struct basic_waitable_timer<chrono::steady_clock>;
}
namespace steady_timer_test
{
void test()
{
    asio::io_context ctx;
    asio::error_code ec;
    bool done = false;
    asio::steady_timer st{ctx, asio::chrono::steady_clock::now()};

    st.async_wait([&](asio::error_code ec_){ec = ec_; done = true;});
    st.notify_one();
    ctx.run();
    ASIO_CHECK(!ec);
    ASIO_CHECK(done);

}
}


ASIO_TEST_SUITE
(
  "steady_timer",
  ASIO_TEST_CASE(steady_timer_test::test)
)
