//
// detail/codecvt.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~
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
#include "asio/detail/codecvt.hpp"

#include "../unit_test.hpp"

namespace codecvt
{
void test_codecvt()
{
  auto end = [](const auto * c){return c + std::char_traits<std::remove_pointer_t<decltype(c)>>::length(c);};

  const char * in = "test-input";

  const wchar_t  * win_t  = L"test-input";
  const char16_t * in16_t = u"test-input";
  const char32_t * in32_t = U"test-input";

#if defined(ASIO_HAS_CHAR8_T)
  const char8_t in8_t[] = u8"test-input";
#endif

  ASIO_CHECK(asio::detail::convert_chars(in,     end(in),      ' ') ==  in);
  ASIO_CHECK(asio::detail::convert_chars(win_t,  end(win_t),  L' ') == win_t);

  ASIO_CHECK(asio::detail::convert_chars(in,     end(in),    L' ') == win_t);
  ASIO_CHECK(asio::detail::convert_chars(win_t,  end(win_t),  ' ') ==  in);

  ASIO_CHECK(asio::detail::convert_chars(in,      end(in),    u' ') == in16_t);
  ASIO_CHECK(asio::detail::convert_chars(in16_t,  end(in16_t), ' ') == in);

  ASIO_CHECK(asio::detail::convert_chars(in,      end(in),    U' ') == in32_t);
  ASIO_CHECK(asio::detail::convert_chars(in32_t,  end(in32_t), ' ') == in);

  ASIO_CHECK(asio::detail::convert_chars(win_t,   end(win_t),  u' ') ==  in16_t);
  ASIO_CHECK(asio::detail::convert_chars(in16_t,  end(in16_t), L' ') == win_t);

  ASIO_CHECK(asio::detail::convert_chars(win_t,   end(win_t),  U' ') ==  in32_t);
  ASIO_CHECK(asio::detail::convert_chars(in32_t,  end(in32_t), L' ') == win_t);

#if defined(ASIO_HAS_CHAR8_T)
  ASIO_CHECK(asio::detail::convert_chars(in8_t, end(in8_t), u8' ') == in8_t);

  ASIO_CHECK(asio::detail::convert_chars(in8_t, end(in8_t),   ' ') == in);
  ASIO_CHECK(asio::detail::convert_chars(in,    end(in),    u8' ') == in8_t);

  ASIO_CHECK(asio::detail::convert_chars(in8_t, end(in8_t),  L' ') == win_t);
  ASIO_CHECK(asio::detail::convert_chars(win_t, end(win_t), u8' ') ==  in8_t);
#endif

}

}

ASIO_TEST_SUITE
(
"detail/codecvt",
ASIO_TEST_CASE(codecvt::test_codecvt)
)
