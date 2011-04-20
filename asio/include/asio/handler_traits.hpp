//
// handler_traits.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_HANDLER_TRAITS_HPP
#define ASIO_HANDLER_TRAITS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/handler_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename Handler>
struct handler_traits
  : asio::detail::handler_traits<Handler>
{
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_HANDLER_TRAITS_HPP
