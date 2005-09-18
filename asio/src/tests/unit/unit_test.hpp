//
// unit_test.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef UNIT_TEST_HPP
#define UNIT_TEST_HPP

#if defined(__BORLANDC__)

// Prevent use of intrinsic for strcmp.
#include <cstring>
#undef strcmp

// Suppress error about condition always being true.
#pragma option -w-ccc

#endif // defined(__BORLANDC__)

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test_framework.hpp>
using boost::unit_test::test_suite;

#endif // UNIT_TEST_HPP
