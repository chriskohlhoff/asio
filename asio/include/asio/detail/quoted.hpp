//
// detail/quoted.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_QUOTED_HPP
#define ASIO_DETAIL_QUOTED_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_QUOTED)
#include <iomanip>
#endif

namespace asio {

#if defined(ASIO_HAS_QUOTED)
using std::quoted;
#else

template<typename T>
inline T& quoted(T & t) {return t;}

template<typename T>
inline T quoted(T t) {return t;}

#endif


} // namespace asio



#endif // ASIO_DETAIL_QUOTED_HPP
