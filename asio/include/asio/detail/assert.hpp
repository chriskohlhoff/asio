//
// detail/assert.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2020 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_ASSERT_HPP
#define ASIO_DETAIL_ASSERT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_BOOST_ASSERT)
# include <boost/assert.hpp>
#else // defined(ASIO_HAS_BOOST_ASSERT)
# include <cassert>
#endif // defined(ASIO_HAS_BOOST_ASSERT)

#if defined(ASIO_HAS_BOOST_ASSERT)
# define ASIO_ASSERT(expr) BOOST_ASSERT(expr)
#else // defined(ASIO_HAS_BOOST_ASSERT)
# define ASIO_ASSERT(expr) assert(expr)
#endif // defined(ASIO_HAS_BOOST_ASSERT)

#if defined(ASIO_HAS_STATIC_ASSERT)
# define ASIO_STATIC_ASSERT(c, n, m) static_assert((c), m)
#else // defined(ASIO_HAS_STATIC_ASSERT)
# define ASIO_STATIC_ASSERT(c, n, m) \
  typedef char static_assert_ ## n \
    [(c) ? 1 : -1] ASIO_UNUSED_TYPEDEF
#endif // defined(ASIO_HAS_STATIC_ASSERT)

#endif // ASIO_DETAIL_ASSERT_HPP
