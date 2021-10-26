//
// pid.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/pid.hpp"

#include "unit_test.hpp"

void check_pid()
{
    ASIO_CHECK(asio::current_pid() != static_cast<asio::pid_type>(0));
}

ASIO_TEST_SUITE
(
  "associator",
  ASIO_TEST_CASE(check_pid)
)
