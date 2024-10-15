//
// detail/filesystem.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_FILESYSTEM_HPP
#define ASIO_DETAIL_FILESYSTEM_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_FILESYSTEM)

#if defined(ASIO_HAS_STD_FILESYSTEM)
# include <filesystem>
#elif defined(ASIO_HAS_STD_EXPERIMENTAL_FILESYSTEM)
# include <experimental/filesystem>
#else // defined(ASIO_HAS_STD_EXPERIMENTAL_STRING_VIEW)
# error ASIO_HAS_FILESYSTEM is set but no filesystem library is available
#endif // defined(ASIO_HAS_STD_EXPERIMENTAL_STRING_VIEW)

namespace asio {

#if defined(ASIO_HAS_STD_FILESYSTEM)
namespace filesystem = std::filesystem;
#elif defined(ASIO_HAS_STD_EXPERIMENTAL_FILESYSTEM)
namespace filesystem = std::experimental::filesystem;
#endif // defined(ASIO_HAS_STD_EXPERIMENTAL_FILESYSTEM)

    } // namespace asio

#endif




#endif // ASIO_DETAIL_FILESYSTEM_HPP
