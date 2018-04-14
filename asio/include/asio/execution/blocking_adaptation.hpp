//
// execution/blocking_adaptation.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_BLOCKING_ADAPTATION_HPP
#define ASIO_EXECUTION_BLOCKING_ADAPTATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution/detail/enumeration.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct blocking_adaptation_t :
  detail::enumeration<blocking_adaptation_t, 2, false>
{
  using detail::enumeration<blocking_adaptation_t, 2, false>::enumeration;

  using allowed_t = enumerator<0>;
  using disallowed_t = enumerator<1>;

  static constexpr allowed_t allowed{};
  static constexpr disallowed_t disallowed{};
};

constexpr blocking_adaptation_t blocking_adaptation{};
inline constexpr blocking_adaptation_t::allowed_t blocking_adaptation_t::allowed;
inline constexpr blocking_adaptation_t::disallowed_t blocking_adaptation_t::disallowed;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_BLOCKING_ADAPTATION_HPP
