//
// detail/regex_fwd.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REGEX_FWD_HPP
#define ASIO_DETAIL_REGEX_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/regex_fwd.hpp>
#include <boost/regex/v4/match_flags.hpp>
#include "asio/detail/pop_options.hpp"

namespace boost {

template <class BidiIterator>
class sub_match;

template <class BidiIterator, class Allocator>
class match_results;

} // namespace boost

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REGEX_FWD_HPP
