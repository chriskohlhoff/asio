//
// execution/outstanding_work.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_OUTSTANDING_WORK_HPP
#define ASIO_EXECUTION_OUTSTANDING_WORK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution/detail/enumeration.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct outstanding_work_t :
  detail::enumeration<outstanding_work_t, 2>
{
  using detail::enumeration<outstanding_work_t, 2>::enumeration;

  using untracked_t = enumerator<0>;
  using tracked_t = enumerator<1>;

  static constexpr untracked_t untracked{};
  static constexpr tracked_t tracked{};
};

constexpr outstanding_work_t outstanding_work{};
inline constexpr outstanding_work_t::untracked_t outstanding_work_t::untracked;
inline constexpr outstanding_work_t::tracked_t outstanding_work_t::tracked;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_OUTSTANDING_WORK_HPP
