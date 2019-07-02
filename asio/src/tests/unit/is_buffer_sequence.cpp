//
// is_buffer_sequence.cpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2019 Alexander Karzhenkov
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//#define ASIO_DISABLE_DECLTYPE

#include "asio/buffer.hpp"
#include "unit_test.hpp"

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

void run()
{
  ASIO_CHECK(!is_mutable_buffer_sequence<A1>::value);
  ASIO_CHECK(!is_mutable_buffer_sequence<B1>::value);
}

} // namespace

ASIO_TEST_SUITE
(
  "is_buffer_sequence",
  ASIO_TEST_CASE(run)
)
