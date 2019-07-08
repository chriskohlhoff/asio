//
// is_buffer_sequence.cpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2019 Alexander Karzhenkov
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <vector>
#include "asio/buffer.hpp"
#include "unit_test.hpp"

#ifdef ASIO_HAS_DECLTYPE
# define ASIO_HAS_DECLTYPE_MSG "ASIO_HAS_DECLTYPE is defined"
# define ASIO_HAS_DECLTYPE_FLAG true
#else
# define ASIO_HAS_DECLTYPE_MSG "ASIO_HAS_DECLTYPE is not defined"
# define ASIO_HAS_DECLTYPE_FLAG false
#endif

using namespace asio;

namespace case1 {

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

struct B2
{
  typedef mutable_buffer value_type;
  typedef void const_iterator; // (!)

  const mutable_buffer* begin() const;
  const mutable_buffer* end() const;
};

struct B3
{
  typedef mutable_buffer value_type;
  typedef const mutable_buffer* const_iterator;

  int begin; // (!)
  const mutable_buffer* end() const;
};

struct C1
{
  typedef mutable_buffer value_type;
  typedef const mutable_buffer* const_iterator;

  const mutable_buffer* begin() const;
  int end() const; // (!)
};

void run()
{
  ASIO_TEST_IOSTREAM << ASIO_HAS_DECLTYPE_MSG << std::endl;

  ASIO_CHECK(!is_mutable_buffer_sequence<A1>::value);
  ASIO_CHECK(!is_mutable_buffer_sequence<B1>::value);
  ASIO_CHECK( is_mutable_buffer_sequence<X1>::value);
  ASIO_CHECK( is_mutable_buffer_sequence<B2>::value == ASIO_HAS_DECLTYPE_FLAG);
  ASIO_CHECK(!is_mutable_buffer_sequence<B3>::value);
  ASIO_CHECK(!is_mutable_buffer_sequence<C1>::value);

  ASIO_CHECK(!is_mutable_buffer_sequence<void>::value);
  ASIO_CHECK(!is_const_buffer_sequence<void>::value);

  ASIO_CHECK(!is_mutable_buffer_sequence<int>::value);
  ASIO_CHECK(!is_const_buffer_sequence<int>::value);

  ASIO_CHECK( is_mutable_buffer_sequence<mutable_buffer>::value);
  ASIO_CHECK( is_const_buffer_sequence<mutable_buffer>::value);

  ASIO_CHECK(!is_mutable_buffer_sequence<const_buffer>::value);
  ASIO_CHECK( is_const_buffer_sequence<const_buffer>::value);

  ASIO_CHECK( is_mutable_buffer_sequence<std::vector<mutable_buffer> >::value);
  ASIO_CHECK( is_const_buffer_sequence<std::vector<mutable_buffer> >::value);

  ASIO_CHECK(!is_mutable_buffer_sequence<std::vector<const_buffer> >::value);
  ASIO_CHECK( is_const_buffer_sequence<std::vector<const_buffer> >::value);
}

} // namespace case1

namespace case2 {
 
template <typename T>
struct S : dynamic_vector_buffer<char, std::allocator<char> >
{
  T size() const;
};

void run()
{
  ASIO_CHECK( is_dynamic_buffer<S<std::size_t> >::value);
  ASIO_CHECK(!is_dynamic_buffer<S<void> >::value);
}

} // namespace case2

ASIO_TEST_SUITE
(
  "is_buffer_sequence",
  ASIO_TEST_CASE(case1::run)
  ASIO_TEST_CASE(case2::run)
)
