//
// tss_bool.hpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TSS_BOOL_HPP
#define ASIO_DETAIL_TSS_BOOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/posix_tss_bool.hpp"
#include "asio/detail/win_tss_bool.hpp"

namespace asio {
namespace detail {

#if defined(_WIN32)
typedef win_tss_bool tss_bool;
#else
typedef posix_tss_bool tss_bool;
#endif

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TSS_BOOL_HPP
