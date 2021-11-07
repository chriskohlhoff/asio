//
// process/this_process/detail/environment_win.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_ENVIRONMENT_WIN_HPP
#define ASIO_DETAIL_IMPL_ENVIRONMENT_WIN_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <algorithm>
#include <cwctype>
#include <cstring>
#include <shellapi.h>

#include "asio/cstring_view.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"


namespace asio
{
namespace environment
{
namespace detail
{

std::basic_string<char_type, value_char_traits<char_type>> get(
        ASIO_BASIC_CSTRING_VIEW_PARAM(char_type, key_char_traits<char_type>) key,
        error_code & ec)
{
  std::basic_string<char_type, value_char_traits<char_type>> buf;

  std::size_t size = 0u;
  do
  {
    buf.resize(buf.size() + 4096);
    size = ::GetEnvironmentVariableW(key.c_str(), buf.data(), buf.size());
  }
  while (size == buf.size());

  buf.resize(size);

  if (buf.size() == 0)
    ec.assign(::GetLastError(), asio::error::get_system_category());

  return buf;
}

void set(ASIO_BASIC_CSTRING_VIEW_PARAM(char_type,   key_char_traits<char_type>)   key,
         ASIO_BASIC_CSTRING_VIEW_PARAM(char_type, value_char_traits<char_type>) value,
         error_code & ec)
{
  if (!::SetEnvironmentVariableW(key.c_str(), value.c_str()))
    ec.assign(errno, asio::error::get_system_category());
}

void unset(ASIO_BASIC_CSTRING_VIEW_PARAM(char_type, key_char_traits<char_type>) key,
           error_code & ec)
{
  if (!::SetEnvironmentVariableW(key.c_str(), nullptr))
    ec.assign(errno, asio::error::get_system_category());
}


std::basic_string<char, value_char_traits<char>> get(
        ASIO_BASIC_CSTRING_VIEW_PARAM(char, key_char_traits<char>) key,
        error_code & ec)
{
  std::basic_string<char, value_char_traits<char>> buf;

  std::size_t size = 0u;
  do
  {
    buf.resize(buf.size() + 4096);
    size = ::GetEnvironmentVariableA(key.c_str(), buf.data(), buf.size());
  }
  while (size == buf.size());

  buf.resize(size);

  if (buf.size() == 0)
    ec.assign(::GetLastError(), asio::error::get_system_category());

  return buf;
}

void set(ASIO_BASIC_CSTRING_VIEW_PARAM(char,   key_char_traits<char>)   key,
         ASIO_BASIC_CSTRING_VIEW_PARAM(char, value_char_traits<char>) value,
         error_code & ec)
{
  if (!::SetEnvironmentVariableA(key.c_str(), value.c_str()))
    ec.assign(errno, asio::error::get_system_category());
}

void unset(ASIO_BASIC_CSTRING_VIEW_PARAM(char, key_char_traits<char>) key,
           error_code & ec)
{
  if (!::SetEnvironmentVariableA(key.c_str(), nullptr))
    ec.assign(errno, asio::error::get_system_category());
}


native_handle_type load_native_handle() { return ::GetEnvironmentStringsW(); }
void native_handle_deleter::operator()(native_handle_type nh) const
{
    ::FreeEnvironmentStringsW(nh);
}

native_iterator next(native_iterator nh)
{
    while (*nh != L'\0')
        nh++;
    return ++nh;
}


native_iterator find_end(native_handle_type nh)
{
  while ((*nh != L'\0') || (*std::next(nh) != L'\0'))
    nh++;
  return ++ ++nh;
}

#if ASIO_HAS_FILESYSTEM
ASIO_DECL bool is_executable(const asio::filesystem::path & pth, error_code & ec)
{
    return asio::filesystem::is_regular_file(pth, ec) && SHGetFileInfoW(pth.native().c_str(), 0,0,0, SHGFI_EXETYPE);
}
#endif

}
}
}

#endif //ASIO_DETAIL_IMPL_ENVIRONMENT_WIN_HPP
