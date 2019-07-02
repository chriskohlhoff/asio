//
// is_buffer_sequence.cpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2019 Alexander Karzhenkov
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "asio/buffer.hpp"
#include "unit_test.hpp"

#ifdef ASIO_HAS_DECLTYPE
# define ASIO_HAS_DECLTYPE_MSG "ASIO_HAS_DECLTYPE is defined"
#else
# define ASIO_HAS_DECLTYPE_MSG "ASIO_HAS_DECLTYPE is not defined"
#endif

using namespace asio;

namespace {

struct A1
{
  mutable_buffer* begin();

  // no "value_type" type
  // no "const_iterator" type
  // no "end" member function
};

struct B1
{
  typedef mutable_buffer value_type;

  // bad "const_iterator" type
  typedef void const_iterator;

  // no "begin" member function
  // no "end" member function
};

struct X1
{
  typedef mutable_buffer value_type;
  typedef const mutable_buffer* const_iterator;

  const mutable_buffer* begin() const;
  const mutable_buffer* end() const;
};

void run()
{
  ASIO_TEST_IOSTREAM << ASIO_HAS_DECLTYPE_MSG << std::endl;

  ASIO_CHECK(!is_mutable_buffer_sequence<A1>::value);
  ASIO_CHECK(!is_mutable_buffer_sequence<B1>::value);
  ASIO_CHECK( is_mutable_buffer_sequence<X1>::value);
}

} // namespace

ASIO_TEST_SUITE
(
  "is_buffer_sequence",
  ASIO_TEST_CASE(run)
)
