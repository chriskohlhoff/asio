//
// process/environment/detail/environment_posix.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_ENVIRONMENT_POSIX_HPP
#define ASIO_DETAIL_ENVIRONMENT_POSIX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/cstring_view.hpp"
#include "asio/error.hpp"
#include "asio/detail/filesystem.hpp"



#include "asio/detail/push_options.hpp"


namespace asio
{
namespace environment
{

using char_type = char;

template<typename Char>
using key_char_traits = std::char_traits<Char>;

template<typename Char>
using value_char_traits = std::char_traits<Char>;


constexpr char_type equality_sign = '=';
constexpr char_type delimiter = ':';

namespace detail
{
std::basic_string<char_type, value_char_traits<char>>
get(ASIO_BASIC_CSTRING_VIEW_PARAM(char_type, key_char_traits<char_type>) key,
    error_code & ec);

ASIO_DECL void set(ASIO_BASIC_CSTRING_VIEW_PARAM(char_type,   key_char_traits<char_type>)   key,
                   ASIO_BASIC_CSTRING_VIEW_PARAM(char_type, value_char_traits<char_type>) value,
                   error_code & ec);

ASIO_DECL void unset(ASIO_BASIC_CSTRING_VIEW_PARAM(char_type, key_char_traits<char_type>) key,
                     error_code & ec);
}


using native_handle_type   = const char * const *;
using native_iterator = native_handle_type;

namespace detail
{

ASIO_DECL native_handle_type load_native_handle();
struct native_handle_deleter
{
    void operator()(native_handle_type) const {}
};

ASIO_DECL native_iterator next(native_handle_type nh);
ASIO_DECL native_iterator find_end(native_handle_type nh);
inline const char_type * dereference(native_iterator iterator) {return *iterator;}
#if ASIO_HAS_FILESYSTEM
ASIO_DECL bool is_executable(const asio::filesystem::path & pth, error_code & ec);
#endif

}

}
}

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/environment_posix.ipp"
#endif // defined(ASIO_HEADER_ONLY)


#endif