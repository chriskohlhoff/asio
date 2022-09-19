// Copyright (c) 2022 Klemens D. Morgenstern
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/early_complete.hpp"

#include "./unit_test.hpp"

struct dummy_init
{
  template<typename Func>
  auto complete_early(Func && func) -> decltype(std::declval<Func>()(int()));

};

void trait_test()
{
  using t1 = asio::has_early_completion<void(), dummy_init>;
  using t2 = asio::has_early_completion<void(int), dummy_init>;
  ASIO_CHECK(!t1::value);
  ASIO_CHECK(t2::value);
}

ASIO_TEST_SUITE
(
    "early_completion",
    ASIO_TEST_CASE(trait_test)
)
