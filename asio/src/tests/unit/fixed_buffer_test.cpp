//
// fixed_buffer_test.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void fixed_buffer_test()
{
  fixed_buffer<32> fb;
  const fixed_buffer<32>& const_fb = fb;

  UNIT_TEST_CHECK(fb.capacity() == 32);
  UNIT_TEST_CHECK(fb.empty());
  UNIT_TEST_CHECK(fb.size() == 0);
  UNIT_TEST_CHECK(fb.begin() == fb.end());
  UNIT_TEST_CHECK(const_fb.begin() == const_fb.end());

  fb.push('A');

  UNIT_TEST_CHECK(!fb.empty());
  UNIT_TEST_CHECK(fb.size() == 1);
  UNIT_TEST_CHECK(fb.begin() != fb.end());
  UNIT_TEST_CHECK(const_fb.begin() != const_fb.end());
  UNIT_TEST_CHECK(fb.front() == 'A');
  UNIT_TEST_CHECK(const_fb.front() == 'A');
  UNIT_TEST_CHECK(fb.back() == 'A');
  UNIT_TEST_CHECK(const_fb.back() == 'A');

  fb.front() = 'B';

  UNIT_TEST_CHECK(!fb.empty());
  UNIT_TEST_CHECK(fb.size() == 1);
  UNIT_TEST_CHECK(fb.begin() != fb.end());
  UNIT_TEST_CHECK(const_fb.begin() != const_fb.end());
  UNIT_TEST_CHECK(fb.front() == 'B');
  UNIT_TEST_CHECK(const_fb.front() == 'B');
  UNIT_TEST_CHECK(fb.back() == 'B');
  UNIT_TEST_CHECK(const_fb.back() == 'B');

  fb.back() = 'C';

  UNIT_TEST_CHECK(!fb.empty());
  UNIT_TEST_CHECK(fb.size() == 1);
  UNIT_TEST_CHECK(fb.begin() != fb.end());
  UNIT_TEST_CHECK(const_fb.begin() != const_fb.end());
  UNIT_TEST_CHECK(fb.front() == 'C');
  UNIT_TEST_CHECK(const_fb.front() == 'C');
  UNIT_TEST_CHECK(fb.back() == 'C');
  UNIT_TEST_CHECK(const_fb.back() == 'C');

  fb.pop();

  UNIT_TEST_CHECK(fb.empty());
  UNIT_TEST_CHECK(fb.size() == 0);
  UNIT_TEST_CHECK(fb.begin() == fb.end());
  UNIT_TEST_CHECK(const_fb.begin() == const_fb.end());

  fb.push('D', 32);

  UNIT_TEST_CHECK(!fb.empty());
  UNIT_TEST_CHECK(fb.size() == 32);
  UNIT_TEST_CHECK(fb.begin() != fb.end());
  UNIT_TEST_CHECK(const_fb.begin() != const_fb.end());
  UNIT_TEST_CHECK(fb.front() == 'D');
  UNIT_TEST_CHECK(const_fb.front() == 'D');
  UNIT_TEST_CHECK(fb.back() == 'D');
  UNIT_TEST_CHECK(const_fb.back() == 'D');
  for (size_t i = 0; i < fb.size(); ++i)
  {
    UNIT_TEST_CHECK(fb[i] == 'D');
    UNIT_TEST_CHECK(const_fb[i] == 'D');
  }

  fb.front() = 'E';

  UNIT_TEST_CHECK(!fb.empty());
  UNIT_TEST_CHECK(fb.size() == 32);
  UNIT_TEST_CHECK(fb.begin() != fb.end());
  UNIT_TEST_CHECK(const_fb.begin() != const_fb.end());
  UNIT_TEST_CHECK(fb.front() == 'E');
  UNIT_TEST_CHECK(const_fb.front() == 'E');
  UNIT_TEST_CHECK(fb.back() == 'D');
  UNIT_TEST_CHECK(const_fb.back() == 'D');

  fb.pop();

  UNIT_TEST_CHECK(!fb.empty());
  UNIT_TEST_CHECK(fb.size() == 31);
  UNIT_TEST_CHECK(fb.begin() != fb.end());
  UNIT_TEST_CHECK(const_fb.begin() != const_fb.end());
  UNIT_TEST_CHECK(fb.front() == 'D');
  UNIT_TEST_CHECK(const_fb.front() == 'D');
  UNIT_TEST_CHECK(fb.back() == 'D');
  UNIT_TEST_CHECK(const_fb.back() == 'D');
  for (char* p = fb.begin(); p != fb.end(); ++p)
    UNIT_TEST_CHECK(*p == 'D');
  for (const char* cp = const_fb.begin(); cp != const_fb.end(); ++cp)
    UNIT_TEST_CHECK(*cp == 'D');

  fb.pop(31);

  UNIT_TEST_CHECK(fb.empty());
  UNIT_TEST_CHECK(fb.size() == 0);
  UNIT_TEST_CHECK(fb.begin() == fb.end());
  UNIT_TEST_CHECK(const_fb.begin() == const_fb.end());

  fb.resize(16);

  UNIT_TEST_CHECK(!fb.empty());
  UNIT_TEST_CHECK(fb.size() == 16);
  UNIT_TEST_CHECK(fb.begin() != fb.end());
  UNIT_TEST_CHECK(const_fb.begin() != const_fb.end());

  fb.clear();

  UNIT_TEST_CHECK(fb.empty());
  UNIT_TEST_CHECK(fb.size() == 0);
  UNIT_TEST_CHECK(fb.begin() == fb.end());
  UNIT_TEST_CHECK(const_fb.begin() == const_fb.end());
}

UNIT_TEST(fixed_buffer_test)
