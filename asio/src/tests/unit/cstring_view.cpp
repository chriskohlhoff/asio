//
// cstring_view.cpp
// ~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern
//                    (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/cstring_view.hpp"
#include <string>
#include "unit_test.hpp"

namespace asio
{

template class basic_cstring_view<char, std::char_traits<char>>;
template class basic_cstring_view<wchar_t, std::char_traits<wchar_t>>;

using char_type = basic_cstring_view<wchar_t, std::char_traits<wchar_t>>::const_pointer;

void cstring_view_test()
{
  cstring_view null;
  ASIO_CHECK(null.c_str() != nullptr);
  ASIO_CHECK(null.c_str()[0] == '\0');

  auto *c = "foobar";
  cstring_view cv = c;

  ASIO_CHECK(cv.c_str() == c);

  std::string s = "barfoo";

  cstring_view sv = s;
  ASIO_CHECK(sv.c_str() == s.c_str());

  auto cp = sv;
}

}



ASIO_TEST_SUITE
(
"cstring_view",
  ASIO_TEST_CASE(asio::cstring_view_test)

)