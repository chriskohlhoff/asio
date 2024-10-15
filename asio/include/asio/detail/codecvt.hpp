//
// detail/codecvt.hpp
// ~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef ASIO_DETAIL_CODECVT_HPP
#define ASIO_DETAIL_CODECVT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/error.hpp"
#include "asio/error_code.hpp"
#include "asio/detail/throw_error.hpp"
#include <locale>


namespace asio
{
namespace detail
{

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)

ASIO_DECL const std::codecvt< wchar_t, char, std::mbstate_t > & default_codecvt();

#else

inline const std::codecvt< wchar_t, char, std::mbstate_t > & default_codecvt()
{
  return std::use_facet<std::codecvt< wchar_t, char, std::mbstate_t >>(std::locale());
}

#endif

// Needed conversions [char8_t, char16_t, char32_t, wchar_t, char] <-> [char, wchar_t]

// C++20

//std::codecvt<char, char, std::mbstate_t>	identity conversion
//std::codecvt<wchar_t, char, std::mbstate_t>	conversion between the system's native wide and the single-byte narrow character sets

//std::codecvt<char16_t, char8_t, std::mbstate_t>	conversion between UTF-16 and UTF-8 (since C++20)
//std::codecvt<char32_t, char8_t, std::mbstate_t>	conversion between UTF-32 and UTF-8 (since C++20)


// C++17

//std::codecvt<char, char, std::mbstate_t>	identity conversion
//std::codecvt<wchar_t, char, std::mbstate_t>	conversion between the system's native wide and the single-byte narrow character sets

//std::codecvt<char16_t, char, std::mbstate_t>	conversion between UTF-16 and UTF-8 (since C++11)(deprecated in C++20)
//std::codecvt<char32_t, char, std::mbstate_t>	conversion between UTF-32 and UTF-8 (since C++11)(deprecated in C++20)



template< class Traits, class Char, class Alloc = std::allocator<Char>>
inline std::basic_string<Char,Traits,Alloc> convert_chars(
        error_code &,
        const Char * begin,
        const Char * end,
        Char,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  return std::basic_string<Char,Traits,Alloc>(begin, end, alloc);
}


template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char,Traits,Alloc> convert_chars(
        error_code & ec,
        const wchar_t * begin,
        const wchar_t * end,
        char,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
      return {};

  const auto & f = loc == std::locale()
                   ? default_codecvt()
                   : std::use_facet<std::codecvt< wchar_t, char, std::mbstate_t > >(loc)
  ;

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = (end - begin) * 2;
  std::basic_string<char, Traits, Alloc> tmp(len, ' ', alloc);

  auto itr = begin;
  auto out_itr = tmp.data();
  auto e = f.out(mb, begin, end, itr, tmp.data(), tmp.data() + tmp.size(), out_itr);
  ec.assign(e, error::get_codecvt_category());
  tmp.resize(out_itr - tmp.data());
  return tmp;
}


template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<wchar_t,Traits,Alloc> convert_chars(
        error_code & ec,
        const char * begin,
        const char * end,
        wchar_t,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
    return {};

  const auto & f = loc == std::locale()
                   ? default_codecvt()
                   : std::use_facet<std::codecvt< wchar_t, char, std::mbstate_t > >(loc);

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = f.length(mb, begin, end, std::numeric_limits<std::size_t>::max());

  std::basic_string<wchar_t,Traits,Alloc> res(len, L' ', alloc);
  auto itr = begin;
  auto out_itr = res.data();
  auto e = f.in(mb, begin, end, itr, res.data(), res.data() + res.size(), out_itr);
  ec.assign(e, error::get_codecvt_category());
  res.resize(out_itr - res.data());
  return res;
}


#if defined(ASIO_HAS_CHAR8_T)

template< class Traits, class Alloc = std::allocator<char8_t>>
inline std::basic_string<char8_t,Traits,Alloc> convert_chars(
        error_code &,
        const char * begin,
        const char * end,
        char8_t,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  return std::basic_string<char8_t,Traits,Alloc>(begin, end, alloc);
}


template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char,Traits,Alloc> convert_chars(
        error_code &,
        const char8_t * begin,
        const char8_t * end,
        char,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  return std::basic_string<char,Traits,Alloc>(begin, end, alloc);
}

template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<wchar_t,Traits,Alloc> convert_chars(
        error_code & ec,
        const char8_t * begin,
        const char8_t * end,
        wchar_t w,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  return convert_chars<Traits>(ec,
                               reinterpret_cast<const char*>(begin),
                               reinterpret_cast<const char*>(end), w, alloc, loc);
}


template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char8_t,Traits,Alloc> convert_chars(
        error_code & ec,
        const wchar_t * begin,
        const wchar_t * end,
        char8_t,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
    return {};

  const auto & f = loc == std::locale()
                   ? default_codecvt()
                   : std::use_facet<std::codecvt< wchar_t, char, std::mbstate_t > >(loc)
  ;

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = (end - begin) * 2;
  std::basic_string<char8_t, Traits, Alloc> tmp(len, ' ', alloc);

  auto itr = begin;
  auto out_itr = tmp.data();
  auto e = f.out(mb, begin, end, itr,
            reinterpret_cast<char*>(tmp.data()),
            reinterpret_cast<char*>(tmp.data() + tmp.size()),
            reinterpret_cast<char*&>(out_itr));
  ec.assign(e, error::get_codecvt_category());
  tmp.resize(out_itr - tmp.data());

  return tmp;
}

template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char,Traits,Alloc> convert_chars(
        error_code & ec,
        const char16_t * begin,
        const char16_t * end,
        char,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
    return {};

  const auto & f = std::use_facet<std::codecvt< char16_t, char8_t, std::mbstate_t > >(loc);

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = (end - begin) * 2;
  std::basic_string<char, Traits, Alloc> tmp(len, ' ', alloc);

  auto itr = begin;
  auto out_itr = tmp.data();
  auto e = f.out(mb, begin, end, itr,
                 reinterpret_cast<char8_t*>(tmp.data()),
                 reinterpret_cast<char8_t*>(tmp.data() + tmp.size()),
                 reinterpret_cast<char8_t *&>(out_itr));

  ec.assign(e, error::get_codecvt_category());
  tmp.resize(out_itr - tmp.data());

  return tmp;
}

template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char,Traits,Alloc> convert_chars(
        error_code & ec,
        const char32_t * begin,
        const char32_t * end,
        char,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
    return {};

  const auto & f = std::use_facet<std::codecvt<char32_t, char8_t, std::mbstate_t > >(loc);

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = (end - begin) * 4;
  std::basic_string<char, Traits, Alloc> tmp(len, ' ', alloc);

  auto itr = begin;
  auto out_itr = tmp.data();
  auto e = f.out(mb, begin, end, itr,
                 reinterpret_cast<char8_t*>(tmp.data()),
                 reinterpret_cast<char8_t*>(tmp.data() + tmp.size()),
                 reinterpret_cast<char8_t *&>(out_itr));
  ec.assign(e, error::get_codecvt_category());

  tmp.resize(out_itr - tmp.data());

  return tmp;
}


template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char16_t,Traits,Alloc> convert_chars(
        error_code &ec,
        const char * begin,
        const char * end,
        char16_t,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
    return {};

  const auto & f = std::use_facet<std::codecvt< char16_t, char8_t, std::mbstate_t > >(loc);

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = f.length(mb, reinterpret_cast<const char8_t*>(begin),
                                       reinterpret_cast<const char8_t*>(end),
                                   std::numeric_limits<std::size_t>::max());

  std::basic_string<char16_t,Traits,Alloc> res(len, u' ', alloc);
  auto itr = begin;
  auto out_itr = res.data();
  auto e = f.in(mb,
                reinterpret_cast<const char8_t*>(begin),
                reinterpret_cast<const char8_t*>(end),
                reinterpret_cast<const char8_t*&>(itr),
                res.data(), res.data() + res.size(), out_itr);
  ec.assign(e, error::get_codecvt_category());
  res.resize(out_itr - res.data());
  return res;
}


template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char32_t,Traits,Alloc> convert_chars(
        error_code & ec,
        const char * begin,
        const char * end,
        char32_t,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
    return {};

  const auto & f = std::use_facet<std::codecvt< char32_t, char8_t, std::mbstate_t > >(loc)
  ;

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = f.length(mb,
                                   reinterpret_cast<const char8_t*>(begin),
                                   reinterpret_cast<const char8_t*>(end),
                                   std::numeric_limits<std::size_t>::max());

  std::basic_string<char32_t,Traits,Alloc> res(len, U' ', alloc);
  auto itr = begin;
  auto out_itr = res.data();
  auto e = f.in(mb,
                reinterpret_cast<const char8_t*>(begin),
                reinterpret_cast<const char8_t*>(end),
                reinterpret_cast<const char8_t*&>(itr),
                res.data(), res.data() + res.size(), out_itr);
  ec.assign(e, error::get_codecvt_category());

  res.resize(out_itr - res.data());
  return res;
}


#else

template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char,Traits,Alloc> convert_chars(
        error_code & ec,
        const char16_t * begin,
        const char16_t * end,
        char,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
    return {};

  const auto & f = std::use_facet<std::codecvt< char16_t, char, std::mbstate_t > >(loc);

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = (end - begin) * 2;
  std::basic_string<char, Traits, Alloc> tmp(len, ' ', alloc);

  auto itr = begin;
  auto out_itr = tmp.data();
  auto e = f.out(mb, begin, end, itr, tmp.data(), tmp.data() + tmp.size(), out_itr);

  ec.assign(e, error::get_codecvt_category());
  tmp.resize(out_itr - tmp.data());

  return tmp;
}

template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char,Traits,Alloc> convert_chars(
        error_code & ec,
        const char32_t * begin,
        const char32_t * end,
        char,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
    return {};

  const auto & f = std::use_facet<std::codecvt<char32_t, char, std::mbstate_t > >(loc);

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = (end - begin) * 4;
  std::basic_string<char, Traits, Alloc> tmp(len, ' ', alloc);

  auto itr = begin;
  auto out_itr = tmp.data();
  auto e = f.out(mb, begin, end, itr, tmp.data(), tmp.data() + tmp.size(), out_itr);
  ec.assign(e, error::get_codecvt_category());

  tmp.resize(out_itr - tmp.data());

  return tmp;
}


template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char16_t,Traits,Alloc> convert_chars(
        error_code &ec,
        const char * begin,
        const char * end,
        char16_t,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
    return {};

  const auto & f = std::use_facet<std::codecvt< char16_t, char, std::mbstate_t > >(loc);

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = f.length(mb, begin, end, std::numeric_limits<std::size_t>::max());

  std::basic_string<char16_t,Traits,Alloc> res(len, u' ', alloc);
  auto itr = begin;
  auto out_itr = res.data();
  auto e = f.in(mb, begin, end, itr, res.data(), res.data() + res.size(), out_itr);
  ec.assign(e, error::get_codecvt_category());
  res.resize(out_itr - res.data());
  return res;
}


template< class Traits, class Alloc = std::allocator<char>>
inline std::basic_string<char32_t,Traits,Alloc> convert_chars(
        error_code & ec,
        const char * begin,
        const char * end,
        char32_t,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  if (begin == end)
    return {};

  const auto & f = std::use_facet<std::codecvt< char32_t, char, std::mbstate_t > >(loc)
  ;

  std::mbstate_t mb = std::mbstate_t();
  const std::size_t len = f.length(mb, begin, end, std::numeric_limits<std::size_t>::max());

  std::basic_string<char32_t,Traits,Alloc> res(len, U' ', alloc);
  auto itr = begin;
  auto out_itr = res.data();
  auto e = f.in(mb, begin, end, itr, res.data(), res.data() + res.size(), out_itr);
  ec.assign(e, error::get_codecvt_category());

  res.resize(out_itr - res.data());
  return res;
}

#endif

template< class Traits, class Alloc = std::allocator<char16_t>>
inline std::basic_string<char16_t,Traits,Alloc> convert_chars(
        error_code &ec,
        const wchar_t * begin,
        const wchar_t * end,
        char16_t,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  using rebind_alloc = typename std::allocator_traits<Alloc>::template rebind_alloc<char>;
  auto tmp = convert_chars<std::char_traits<char>>(ec, begin, end, ' ', rebind_alloc(alloc), loc);

  if (ec)
    return u"";

  return convert_chars<Traits>(ec, tmp.data(), tmp.data() + tmp.size(), u' ', alloc, loc);
}


template< class Traits, class Alloc = std::allocator<char32_t>>
inline std::basic_string<char32_t,Traits,Alloc> convert_chars(
        error_code  &ec,
        const wchar_t * begin,
        const wchar_t * end,
        char32_t,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  using rebind_alloc = typename std::allocator_traits<Alloc>::template rebind_alloc<char>;
  auto tmp = convert_chars<std::char_traits<char>>(ec, begin, end, ' ', rebind_alloc(alloc), loc);
  if (ec)
    return U"";
  return convert_chars<Traits>(ec, tmp.data(), tmp.data() + tmp.size(), U' ', alloc, loc);
}


template< class Traits, class Alloc = std::allocator<wchar_t>>
inline std::basic_string<wchar_t,Traits,Alloc> convert_chars(
        error_code &ec,
        const char16_t * begin,
        const char16_t * end,
        wchar_t,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  using rebind_alloc = typename std::allocator_traits<Alloc>::template rebind_alloc<char>;
  auto tmp = convert_chars<std::char_traits<char>>(ec, begin, end, ' ', rebind_alloc(alloc), loc);
  if (ec)
    return L"";
  return convert_chars<Traits>(ec, tmp.data(), tmp.data() + tmp.size(), L' ', alloc, loc);
}


template< class Traits, class Alloc = std::allocator<wchar_t>>
inline std::basic_string<wchar_t,Traits,Alloc> convert_chars(
        error_code &ec,
        const char32_t * begin,
        const char32_t * end,
        wchar_t w,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  using rebind_alloc = typename std::allocator_traits<Alloc>::template rebind_alloc<char>;
  auto tmp = convert_chars<std::char_traits<char>>(ec, begin, end, ' ', rebind_alloc(alloc), loc);
  if (ec)
    return L"";
  return convert_chars<Traits>(ec, tmp.data(), tmp.data() + tmp.size(), L' ', alloc, loc);
}



template< class Traits, class CharIn, class CharOut, class Alloc = std::allocator<CharOut>>
inline std::basic_string<CharOut, Traits, Alloc> convert_chars(
        const CharIn * begin,
        const CharIn * end,
        CharOut c,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  error_code ec;
  auto res =  convert_chars<Traits>(ec, begin, end, c, alloc, loc);
  if (ec)
    asio::detail::throw_error(ec, "convert_chars");
  return res;
}

template< class CharIn, class CharOut, class Alloc = std::allocator<CharOut>>
inline std::basic_string<CharOut, std::char_traits<CharOut>, Alloc> convert_chars(
        error_code & ec,
        const CharIn * begin,
        const CharIn * end,
        CharOut c,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  return convert_chars<std::char_traits<CharOut>>(ec, begin, end, c, alloc, loc);
}

template< class CharIn, class CharOut, class Alloc = std::allocator<CharOut>>
inline std::basic_string<CharOut, std::char_traits<CharOut>, Alloc> convert_chars(
        const CharIn * begin,
        const CharIn * end,
        CharOut c,
        const Alloc & alloc = Alloc(),
        const std::locale &loc = std::locale())
{
  error_code ec;
  auto res =  convert_chars<std::char_traits<CharOut>>(ec, begin, end, c, alloc, loc);
  if (ec)
    asio::detail::throw_error(ec, "convert_chars");
  return res;
}



} // detail
} // asio

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/codecvt.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif //ASIO_DETAIL_CODECVT_HPP
