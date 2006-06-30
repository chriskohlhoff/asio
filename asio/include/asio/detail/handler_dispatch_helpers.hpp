//
// handler_dispatch_helpers.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_HANDLER_DISPATCH_HELPERS_HPP
#define ASIO_DETAIL_HANDLER_DISPATCH_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/handler_dispatch_hook.hpp"

// Calls to asio_handler_dispatch must be made from a namespace that does not
// contain any overloads of this function. The asio_handler_dispatch_helpers
// namespace is defined here for that purpose.
namespace asio_handler_dispatch_helpers {

template <typename Handler, typename Context>
inline void dispatch_handler(const Handler& handler, Context* context)
{
  using namespace asio;
  asio_handler_dispatch(handler, context);
}

} // namespace asio_handler_dispatch_helpers

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_HANDLER_DISPATCH_HELPERS_HPP
