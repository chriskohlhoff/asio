//
// execution/bulk_guarantee.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_BULK_GUARANTEE_HPP
#define ASIO_EXECUTION_BULK_GUARANTEE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution/detail/enumeration.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct bulk_guarantee_t :
  detail::enumeration<bulk_guarantee_t, 3>
{
  using detail::enumeration<bulk_guarantee_t, 3>::enumeration;

  using unsequenced_t = enumerator<0>;
  using sequenced_t = enumerator<1>;
  using parallel_t = enumerator<2>;

  static constexpr unsequenced_t unsequenced{};
  static constexpr sequenced_t sequenced{};
  static constexpr parallel_t parallel{};
};

constexpr bulk_guarantee_t bulk_guarantee{};
inline constexpr bulk_guarantee_t::unsequenced_t bulk_guarantee_t::unsequenced;
inline constexpr bulk_guarantee_t::sequenced_t bulk_guarantee_t::sequenced;
inline constexpr bulk_guarantee_t::parallel_t bulk_guarantee_t::parallel;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_BULK_GUARANTEE_HPP
