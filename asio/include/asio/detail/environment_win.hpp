//
// process/environment/detail/environment_win.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_ENVIRONMENT_WIN_HPP
#define ASIO_DETAIL_ENVIRONMENT_WIN_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <algorithm>
#include <cwctype>
#include <cstring>

#include "asio/cstring_view.hpp"
#include "asio/error.hpp"
#include "asio/detail/filesystem.hpp"

#include "asio/detail/push_options.hpp"


namespace asio
{
namespace environment
{

using char_type = wchar_t;
template<typename Char>
struct key_char_traits
{
  typedef Char      char_type;
  typedef int       int_type;
  typedef std::streamoff off_type;
  typedef std::streampos pos_type;
  typedef std::mbstate_t state_type;

  ASIO_CONSTEXPR static char    to_upper(char c)    {return std::toupper(c);}
  ASIO_CONSTEXPR static wchar_t to_upper(wchar_t c) {return std::towupper(c);}

  ASIO_CONSTEXPR static int_type to_upper(int_type i, char )   {return std::toupper(i);}
  ASIO_CONSTEXPR static int_type to_upper(int_type i, wchar_t) {return std::towupper(i);}


  ASIO_CONSTEXPR static
  void assign(char_type& c1, const char_type& c2) ASIO_NOEXCEPT
  {
    c1 = c2;
  }

  ASIO_CONSTEXPR static
  bool eq(char_type c1, char_type c2) ASIO_NOEXCEPT
  {
    return to_upper(c1) == to_upper(c2);
  }

  ASIO_CONSTEXPR static
  bool lt(char_type c1, char_type c2) ASIO_NOEXCEPT
  {
    return to_upper(c1) < to_upper(c2);
  }

  ASIO_CONSTEXPR static
  int compare(const char_type* s1, const char_type* s2, size_t n) ASIO_NOEXCEPT
  {
    auto itrs = std::mismatch(s1, s1 + n, s2, &eq);
    if (itrs.first == (s1 + n))
      return 0;
    auto c1 = to_upper(*itrs.first);
    auto c2 = to_upper(*itrs.second);

    return (c1 < c2 ) ? -1 : 1;
  }

  ASIO_CONSTEXPR static size_t length(const char* s)    ASIO_NOEXCEPT  { return std::strlen(s); }
  ASIO_CONSTEXPR static size_t length(const wchar_t* s) ASIO_NOEXCEPT  { return std::wcslen(s); }

  ASIO_CONSTEXPR static
  const char_type* find(const char_type* s, size_t n, const char_type& a) ASIO_NOEXCEPT
  {
    const char_type u = to_upper(a);
    return std::find_if(s, s + n, [u](char_type c){return to_upper(c) == u;});
  }

  ASIO_CONSTEXPR static
  char_type* move(char_type* s1, const char_type* s2, size_t n) ASIO_NOEXCEPT
  {
    if (s1 < s2)
      return std::move(s2, s2 + n, s1);
    else
      return std::move_backward(s2, s2 + n, s1);
  }

  ASIO_CONSTEXPR static
  char_type* copy(char_type* s1, const char_type* s2, size_t n) ASIO_NOEXCEPT
  {
    return std::copy(s2, s2 + n, s1);
  }

  ASIO_CONSTEXPR static
  char_type* assign(char_type* s, size_t n, char_type a) ASIO_NOEXCEPT
  {
    std::fill(s, s + n, a);
    return s +n;
  }

  ASIO_CONSTEXPR static
  int_type not_eof(int_type c) ASIO_NOEXCEPT
  {
    return eq_int_type(c, eof()) ? ~eof() : c;
  }

  ASIO_CONSTEXPR static
  char_type to_char_type(int_type c) ASIO_NOEXCEPT
  {
    return char_type(c);
  }

  ASIO_CONSTEXPR static
  int_type to_int_type(char c) ASIO_NOEXCEPT
  {
    return int_type((unsigned char)c);
  }

  ASIO_CONSTEXPR static
  int_type to_int_type(wchar_t c) ASIO_NOEXCEPT
  {
    return int_type((wchar_t)c);
  }

  ASIO_CONSTEXPR static
  bool eq_int_type(int_type c1, int_type c2) ASIO_NOEXCEPT
  {
    return to_upper(c1, char_type()) == to_upper(c2, char_type());
  }

  ASIO_CONSTEXPR static inline int_type eof() ASIO_NOEXCEPT
  {
    return int_type(EOF);
  }
};


template<typename Char>
using value_char_traits = std::char_traits<Char>;

ASIO_CONSTEXPR static char_type equality_sign = L'=';
ASIO_CONSTEXPR static char_type delimiter = L';';

using native_handle_type   = wchar_t*;
using native_iterator = const wchar_t*;

namespace detail
{

ASIO_DECL std::basic_string<wchar_t, value_char_traits<wchar_t>> get(
        ASIO_BASIC_CSTRING_VIEW_PARAM(wchar_t, key_char_traits<wchar_t>) key,
        error_code & ec);

ASIO_DECL void set(ASIO_BASIC_CSTRING_VIEW_PARAM(wchar_t,   key_char_traits<wchar_t>)   key,
                   ASIO_BASIC_CSTRING_VIEW_PARAM(wchar_t, value_char_traits<wchar_t>) value,
                   error_code & ec);

ASIO_DECL void unset(ASIO_BASIC_CSTRING_VIEW_PARAM(wchar_t, key_char_traits<wchar_t>) key,
                     error_code & ec);


ASIO_DECL std::basic_string<char, value_char_traits<char>> get(
        ASIO_BASIC_CSTRING_VIEW_PARAM(char, key_char_traits<char>) key,
        error_code & ec);

ASIO_DECL void set(ASIO_BASIC_CSTRING_VIEW_PARAM(char,   key_char_traits<char>)   key,
                   ASIO_BASIC_CSTRING_VIEW_PARAM(char, value_char_traits<char>) value,
                   error_code & ec);

ASIO_DECL void unset(ASIO_BASIC_CSTRING_VIEW_PARAM(char, key_char_traits<char>) key,
                     error_code & ec);

inline native_handle_type load_native_handle();
struct native_handle_deleter
{
  ASIO_DECL void operator()(native_handle_type nh) const;

};

inline const char_type * dereference(native_iterator iterator) {return iterator;}
ASIO_DECL native_iterator next(native_handle_type nh);
ASIO_DECL native_iterator find_end(native_handle_type nh);

#if ASIO_HAS_FILESYSTEM
ASIO_DECL bool is_executable(const asio::filesystem::path & pth, error_code & ec);
#endif

}
}
}

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/environment_win.ipp"
#endif // defined(ASIO_HEADER_ONLY)


#endif //ASIO_DETAIL_ENVIRONMENT_WIN_HPP
