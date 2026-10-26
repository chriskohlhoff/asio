//
// is_shallow_mutable_sequence.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/is_shallow_mutable_sequence.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>
#include "unit_test.hpp"

#if defined(ASIO_HAS_STD_RANGES)
# include <ranges>
# include <span>
#endif // defined(ASIO_HAS_STD_RANGES)

class shared_sizes
{
public:
  shared_sizes(std::size_t n)
    : body_(std::make_shared<std::vector<std::size_t>>(n))
  {
  }

  std::vector<std::size_t>::iterator begin() const
  {
    return body_->begin();
  }

  std::vector<std::size_t>::iterator end() const
  {
    return body_->end();
  }

private:
  std::shared_ptr<std::vector<std::size_t>> body_;
};

namespace asio {

template <>
struct is_shallow_mutable_sequence<shared_sizes> : true_type
{
};

} // namespace asio

void is_shallow_mutable_sequence_test()
{
  typedef std::array<std::size_t, 4> size_array;
  typedef std::array<std::reference_wrapper<std::size_t>, 4> ref_size_array;
  typedef std::array<std::reference_wrapper<const std::size_t>, 4>
    const_ref_size_array;

  ASIO_CHECK(!!asio::is_shallow_mutable_sequence<
      std::reference_wrapper<size_array> >::value);

  ASIO_CHECK(!!asio::is_shallow_mutable_sequence<
      std::reference_wrapper<std::vector<std::size_t> > >::value);

  ASIO_CHECK(!!asio::is_shallow_mutable_sequence<ref_size_array>::value);

  ASIO_CHECK(!!asio::is_shallow_mutable_sequence<
      std::vector<std::reference_wrapper<std::size_t> > >::value);

  ASIO_CHECK(!asio::is_shallow_mutable_sequence<
      std::reference_wrapper<const size_array> >::value);

  ASIO_CHECK(!asio::is_shallow_mutable_sequence<const_ref_size_array>::value);

  ASIO_CHECK(!asio::is_shallow_mutable_sequence<size_array>::value);

  ASIO_CHECK(!asio::is_shallow_mutable_sequence<
      std::vector<std::size_t> >::value);

  ASIO_CHECK(!asio::is_shallow_mutable_sequence<int>::value);

  ASIO_CHECK(!!asio::is_shallow_mutable_sequence<shared_sizes>::value);

#if defined(ASIO_HAS_STD_RANGES)
  ASIO_CHECK(!!asio::is_shallow_mutable_sequence<
      std::span<std::size_t> >::value);

  ASIO_CHECK(!asio::is_shallow_mutable_sequence<
      std::span<const std::size_t> >::value);

  ASIO_CHECK(!!asio::is_shallow_mutable_sequence<
      std::ranges::subrange<std::size_t*> >::value);
#endif // defined(ASIO_HAS_STD_RANGES)
}

ASIO_TEST_SUITE
(
  "is_shallow_mutable_sequence",
  ASIO_TEST_CASE(is_shallow_mutable_sequence_test)
)
