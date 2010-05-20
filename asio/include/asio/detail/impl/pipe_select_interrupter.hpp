//
// detail/impl/pipe_select_interrupter.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_PIPE_SELECT_INTERRUPTER_HPP
#define ASIO_DETAIL_IMPL_PIPE_SELECT_INTERRUPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

inline int pipe_select_interrupter::read_descriptor() const
{
  return read_descriptor_;
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#endif // ASIO_DETAIL_IMPL_PIPE_SELECT_INTERRUPTER_HPP
