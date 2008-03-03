//
// openssl_types.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2005-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_OPENSSL_TYPES_HPP
#define ASIO_SSL_DETAIL_OPENSSL_TYPES_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"
#include <openssl/conf.h>
#include <openssl/ssl.h>
#include <openssl/engine.h>
#include <openssl/err.h>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_OPENSSL_TYPES_HPP
