//
// execution/relationship.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_RELATIONSHIP_HPP
#define ASIO_EXECUTION_RELATIONSHIP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution/detail/enumeration.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct relationship_t :
  detail::enumeration<relationship_t, 2, false>
{
  using detail::enumeration<relationship_t, 2, false>::enumeration;

  using continuation_t = enumerator<0>;
  using fork_t = enumerator<1>;

  static constexpr continuation_t continuation{};
  static constexpr fork_t fork{};
};

constexpr relationship_t relationship{};
inline constexpr relationship_t::continuation_t relationship_t::continuation;
inline constexpr relationship_t::fork_t relationship_t::fork;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_RELATIONSHIP_HPP
