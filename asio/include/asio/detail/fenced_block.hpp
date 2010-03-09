//
// fenced_block.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_FENCED_BLOCK_HPP
#define ASIO_DETAIL_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#if 1//!defined(BOOST_HAS_THREADS)
# include "asio/detail/null_fenced_block.hpp"
#elif defined(__MACH__) && defined(__APPLE__)
# include "asio/detail/macos_fenced_block.hpp"
#elif defined(__sun)
# include "asio/detail/solaris_fenced_block.hpp"
#elif defined(__GNUC__) \
  && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4))
# include "asio/detail/gcc_fenced_block.hpp"
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
# include "asio/detail/gcc_x86_fenced_block.hpp"
#elif defined(BOOST_WINDOWS)
# include "asio/detail/win_fenced_block.hpp"
#else
# include "asio/detail/null_fenced_block.hpp"
#endif

namespace asio {
namespace detail {

#if 1//!defined(BOOST_HAS_THREADS)
typedef null_fenced_block fenced_block;
#elif defined(__MACH__) && defined(__APPLE__)
typedef macos_fenced_block fenced_block;
#elif defined(__sun)
typedef solaris_fenced_block fenced_block;
#elif defined(__GNUC__) \
  && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4))
typedef gcc_fenced_block fenced_block;
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
typedef gcc_x86_fenced_block fenced_block;
#elif defined(BOOST_WINDOWS)
typedef win_fenced_block fenced_block;
#else
typedef null_fenced_block fenced_block;
#endif

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_FENCED_BLOCK_HPP
