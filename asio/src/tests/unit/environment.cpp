//
// environment.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/error_code.hpp"
#include "unit_test.hpp"


#include "asio/environment.hpp"
#include <algorithm>
#include "unit_test.hpp"

namespace environment
{
void simple_test()
try
{
    namespace env = asio::environment;
    for (auto [key, value] : env::view())
    {
      if (key.empty())
        continue;
      const auto kenv = ::getenv(key.string().c_str());;
      ASIO_CHECK_MESSAGE(kenv != nullptr, key);
      std::string ptr = kenv;
      env::value v = ptr;
      ASIO_CHECK(v == ptr);
      env::key_view kv = ptr;
      static_assert(asio::is_constructible<env::key_view, std::string>::value);
      static_assert(asio::is_convertible<std::string, env::key_view>::value);
      ASIO_CHECK(ptr == value);
      ASIO_CHECK(env::get(key) == ptr);
    }

    {
        auto v = env::view();
        auto itr = std::find_if(v.begin(), v.end(), [](env::key_value_pair kvp) {return kvp.key_view() == "ASIO_ENV_TEST";});
        ASIO_CHECK(itr == v.end());
        asio::error_code ec;
        ASIO_CHECK(env::get("ASIO_ENV_TEST", ec).empty());
    }
    {
        asio::error_code ec;
        env::set("ASIO_ENV_TEST", "123", ec);
        ASIO_CHECK_MESSAGE(!ec, ec.message());
        auto v = env::view();
        auto itr = std::find_if(v.begin(), v.end(), [](env::key_value_pair kvp) {return kvp.key_view() == "ASIO_ENV_TEST";});
        ASIO_CHECK(itr != v.end());

        using  std::operator""s;

      ASIO_CHECK(*itr == L"ASIO_ENV_TEST=123"s);
      ASIO_CHECK(*itr == "ASIO_ENV_TEST=123"s);
      ASIO_CHECK(itr->key_view() == L"ASIO_ENV_TEST"s);
      ASIO_CHECK(itr->key_view() == "ASIO_ENV_TEST"s);
      ASIO_CHECK(itr->value_view() == L"123"s);
      ASIO_CHECK(itr->value_view() == "123"s);
      ASIO_CHECK(env::get("ASIO_ENV_TEST") == "123"s);

    }
    {
        env::unset("ASIO_ENV_TEST");
        auto v = env::view();
        auto itr = std::find_if(v.begin(), v.end(), [](env::key_value_pair kvp) {return kvp.key_view() == "ASIO_ENV_TEST";});
        ASIO_CHECK(itr == v.end());
        asio::error_code ec;

        ASIO_CHECK(env::get("ASIO_ENV_TEST", ec).empty());
    }

#if defined(ASIO_WINDOWS)
    ASIO_CHECK(!asio::environment::find_executable("cmd").empty());
#else
  ASIO_CHECK((asio::environment::find_executable("ls") == "/bin/ls")
         ||  (asio::environment::find_executable("ls") == "/usr/bin/ls"));
#endif

}
catch (std::exception & ex)
{
    ASIO_CHECK_MESSAGE(false, ex.what());
}

}


ASIO_TEST_SUITE
(
        "environment",
        ASIO_TEST_CASE(environment::simple_test)
)