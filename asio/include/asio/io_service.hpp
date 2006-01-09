//
// io_service.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IO_SERVICE_HPP
#define ASIO_IO_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_io_service.hpp"

namespace asio {

/// Typedef for typical usage of io_service.
typedef basic_io_service<> io_service;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IO_SERVICE_HPP
