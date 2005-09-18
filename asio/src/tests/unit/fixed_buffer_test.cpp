//
// fixed_buffer_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/fixed_buffer.hpp"

#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void fixed_buffer_test()
{
  fixed_buffer<32> fb;
  const fixed_buffer<32>& const_fb = fb;

  BOOST_CHECK(fb.capacity() == 32);
  BOOST_CHECK(fb.empty());
  BOOST_CHECK(fb.size() == 0);
  BOOST_CHECK(fb.begin() == fb.end());
  BOOST_CHECK(const_fb.begin() == const_fb.end());

  fb.push('A');

  BOOST_CHECK(!fb.empty());
  BOOST_CHECK(fb.size() == 1);
  BOOST_CHECK(fb.begin() != fb.end());
  BOOST_CHECK(const_fb.begin() != const_fb.end());
  BOOST_CHECK(fb.front() == 'A');
  BOOST_CHECK(const_fb.front() == 'A');
  BOOST_CHECK(fb.back() == 'A');
  BOOST_CHECK(const_fb.back() == 'A');

  fb.front() = 'B';

  BOOST_CHECK(!fb.empty());
  BOOST_CHECK(fb.size() == 1);
  BOOST_CHECK(fb.begin() != fb.end());
  BOOST_CHECK(const_fb.begin() != const_fb.end());
  BOOST_CHECK(fb.front() == 'B');
  BOOST_CHECK(const_fb.front() == 'B');
  BOOST_CHECK(fb.back() == 'B');
  BOOST_CHECK(const_fb.back() == 'B');

  fb.back() = 'C';

  BOOST_CHECK(!fb.empty());
  BOOST_CHECK(fb.size() == 1);
  BOOST_CHECK(fb.begin() != fb.end());
  BOOST_CHECK(const_fb.begin() != const_fb.end());
  BOOST_CHECK(fb.front() == 'C');
  BOOST_CHECK(const_fb.front() == 'C');
  BOOST_CHECK(fb.back() == 'C');
  BOOST_CHECK(const_fb.back() == 'C');

  fb.pop();

  BOOST_CHECK(fb.empty());
  BOOST_CHECK(fb.size() == 0);
  BOOST_CHECK(fb.begin() == fb.end());
  BOOST_CHECK(const_fb.begin() == const_fb.end());

  fb.push('D', 32);

  BOOST_CHECK(!fb.empty());
  BOOST_CHECK(fb.size() == 32);
  BOOST_CHECK(fb.begin() != fb.end());
  BOOST_CHECK(const_fb.begin() != const_fb.end());
  BOOST_CHECK(fb.front() == 'D');
  BOOST_CHECK(const_fb.front() == 'D');
  BOOST_CHECK(fb.back() == 'D');
  BOOST_CHECK(const_fb.back() == 'D');
  for (size_t i = 0; i < fb.size(); ++i)
  {
    BOOST_CHECK(fb[i] == 'D');
    BOOST_CHECK(const_fb[i] == 'D');
  }

  fb.front() = 'E';

  BOOST_CHECK(!fb.empty());
  BOOST_CHECK(fb.size() == 32);
  BOOST_CHECK(fb.begin() != fb.end());
  BOOST_CHECK(const_fb.begin() != const_fb.end());
  BOOST_CHECK(fb.front() == 'E');
  BOOST_CHECK(const_fb.front() == 'E');
  BOOST_CHECK(fb.back() == 'D');
  BOOST_CHECK(const_fb.back() == 'D');

  fb.pop();

  BOOST_CHECK(!fb.empty());
  BOOST_CHECK(fb.size() == 31);
  BOOST_CHECK(fb.begin() != fb.end());
  BOOST_CHECK(const_fb.begin() != const_fb.end());
  BOOST_CHECK(fb.front() == 'D');
  BOOST_CHECK(const_fb.front() == 'D');
  BOOST_CHECK(fb.back() == 'D');
  BOOST_CHECK(const_fb.back() == 'D');
  for (char* p = fb.begin(); p != fb.end(); ++p)
    BOOST_CHECK(*p == 'D');
  for (const char* cp = const_fb.begin(); cp != const_fb.end(); ++cp)
    BOOST_CHECK(*cp == 'D');

  fb.pop(31);

  BOOST_CHECK(fb.empty());
  BOOST_CHECK(fb.size() == 0);
  BOOST_CHECK(fb.begin() == fb.end());
  BOOST_CHECK(const_fb.begin() == const_fb.end());

  fb.resize(16);

  BOOST_CHECK(!fb.empty());
  BOOST_CHECK(fb.size() == 16);
  BOOST_CHECK(fb.begin() != fb.end());
  BOOST_CHECK(const_fb.begin() != const_fb.end());

  fb.clear();

  BOOST_CHECK(fb.empty());
  BOOST_CHECK(fb.size() == 0);
  BOOST_CHECK(fb.begin() == fb.end());
  BOOST_CHECK(const_fb.begin() == const_fb.end());
}

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("fixed_buffer");
  test->add(BOOST_TEST_CASE(&fixed_buffer_test));
  return test;
}
