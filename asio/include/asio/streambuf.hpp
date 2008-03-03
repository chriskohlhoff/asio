//
// streambuf.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_STREAMBUF_HPP
#define ASIO_STREAMBUF_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_streambuf.hpp"

namespace asio {

/// Typedef for the typical usage of basic_streambuf.
typedef basic_streambuf<> streambuf;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_STREAMBUF_HPP
