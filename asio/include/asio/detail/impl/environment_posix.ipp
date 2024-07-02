//
// process/this_process/detail/environment_posix.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_ENVIRONMENT_POSIX_HPP
#define ASIO_DETAIL_IMPL_ENVIRONMENT_POSIX_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <cstring>

#include "asio/cstring_view.hpp"
#include "asio/error.hpp"
#include <unistd.h>

#include "asio/detail/push_options.hpp"

namespace asio
{
namespace environment
{
namespace detail
{


std::basic_string<char_type, value_char_traits<char>> get(
        ASIO_BASIC_CSTRING_VIEW_PARAM(char_type, key_char_traits<char_type>) key,
        error_code & ec)
{
    auto res = ::getenv(key.c_str());
    if (res == nullptr)
    {
        ec.assign(errno, asio::error::get_system_category());
        return {};
    }
    return res;
}

void set(ASIO_BASIC_CSTRING_VIEW_PARAM(char_type,   key_char_traits<char_type>)   key,
         ASIO_BASIC_CSTRING_VIEW_PARAM(char_type, value_char_traits<char_type>) value,
                error_code & ec)
{
    if (::setenv(key.c_str(), value.c_str(), true))
        ec.assign(errno, asio::error::get_system_category());
}

void unset(ASIO_BASIC_CSTRING_VIEW_PARAM(char_type, key_char_traits<char_type>) key,
                  error_code & ec)
{
    if (::unsetenv(key.c_str()))
        ec.assign(errno, asio::error::get_system_category());
}


native_handle_type load_native_handle() { return ::environ; }


native_iterator next(native_iterator nh)
{
    return nh + 1;
}

native_iterator find_end(native_handle_type nh)
{
    while (*nh != nullptr)
        nh++;
    return nh;
}
#if ASIO_HAS_FILESYSTEM
ASIO_DECL bool is_executable(const asio::filesystem::path & p, error_code & ec)
{
    return asio::filesystem::is_regular_file(p, ec) && (::access(p.c_str(), X_OK) == 0);
}
#endif

}
}
}

#endif //ASIO_DETAIL_IMPL_ENVIRONMENT_POSIX_HPP
