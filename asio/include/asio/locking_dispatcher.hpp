//
// locking_dispatcher.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_LOCKING_DISPATCHER_HPP
#define ASIO_LOCKING_DISPATCHER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_locking_dispatcher.hpp"
#include "asio/io_service.hpp"

namespace asio {

/// Typedef for the typical usage of locking_dispatcher.
typedef basic_locking_dispatcher<io_service> locking_dispatcher;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_LOCKING_DISPATCHER_HPP
