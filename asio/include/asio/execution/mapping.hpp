//
// execution/mapping.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_MAPPING_HPP
#define ASIO_EXECUTION_MAPPING_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution/detail/enumeration.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct mapping_t :
  detail::enumeration<mapping_t, 3, false>
{
  using detail::enumeration<mapping_t, 3, false>::enumeration;

  using other_t = enumerator<0>;
  using thread_t = enumerator<1>;
  using new_thread_t = enumerator<2>;

  static constexpr other_t other{};
  static constexpr thread_t thread{};
  static constexpr new_thread_t new_thread{};
};

constexpr mapping_t mapping{};
inline constexpr mapping_t::other_t mapping_t::other;
inline constexpr mapping_t::thread_t mapping_t::thread;
inline constexpr mapping_t::new_thread_t mapping_t::new_thread;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_MAPPING_HPP
