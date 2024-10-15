//
// experimental/cstring_view.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern
//                    (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef ASIO_CSTRING_VIEW_HPP
#define ASIO_CSTRING_VIEW_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/detail/string_view.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/push_options.hpp"

namespace asio
{

#if defined(ASIO_HAS_STRING_VIEW)

//based on http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1402r0.pdf
///cstring_view is simply a forwarding adapter around string_view.
/** Operations are delegated to a private member string_view, with some minor modifications
    to the interface to maintain the null-terminated class invariant. */
template<typename CharT, typename Traits = std::char_traits<CharT>>
struct basic_cstring_view
{
  using value_type             = CharT;
  using traits_type            = Traits;

  ASIO_CONSTEXPR basic_cstring_view() : view_(null_char_(value_type{})) {};
  ASIO_CONSTEXPR basic_cstring_view(std::nullptr_t) = delete;

  ASIO_CONSTEXPR basic_cstring_view( const value_type* s ) : view_(s) {}

  template<typename Source,
          typename =
          typename enable_if<
                  is_same<const value_type,
                          typename remove_pointer<decltype(std::declval<Source>().c_str())>::type
                  >::value>::type>
  ASIO_CONSTEXPR basic_cstring_view(Source && src) : view_(src.c_str()) {}

  ASIO_CONSTEXPR typename std::basic_string_view<value_type, Traits>::const_pointer c_str() const ASIO_NOEXCEPT
  {
    return this->data();
  }

  using string_view_type = basic_string_view<value_type, Traits>;
  operator string_view_type() const {return view_;}

  using pointer                =       CharT *;
  using const_pointer          = const CharT *;
  using reference              =       CharT &;
  using const_reference        = const CharT &;
  using const_iterator         = const_pointer;
  using iterator               = const_iterator;
  using const_reverse_iterator = typename std::reverse_iterator<const_iterator>;
  using reverse_iterator       = typename std::reverse_iterator<iterator>;
  using size_type              = std::size_t;
  using difference_type        = std::ptrdiff_t;

  static ASIO_CONSTEXPR size_type npos = -1;

  ASIO_CONSTEXPR const_iterator begin()  const ASIO_NOEXCEPT {return view_;};
  ASIO_CONSTEXPR const_iterator end()    const ASIO_NOEXCEPT {return view_ + length();};
  ASIO_CONSTEXPR const_iterator cbegin() const ASIO_NOEXCEPT {return view_;};
  ASIO_CONSTEXPR const_iterator cend()   const ASIO_NOEXCEPT {return view_ + length();};
  ASIO_CONSTEXPR const_reverse_iterator rbegin()  const ASIO_NOEXCEPT {return std::make_reverse_iterator(view_ + length());};
  ASIO_CONSTEXPR const_reverse_iterator rend()    const ASIO_NOEXCEPT {return std::make_reverse_iterator(view_);};
  ASIO_CONSTEXPR const_reverse_iterator crbegin() const ASIO_NOEXCEPT {return std::make_reverse_iterator(view_ + length());};
  ASIO_CONSTEXPR const_reverse_iterator crend()   const ASIO_NOEXCEPT {return std::make_reverse_iterator(view_);};

  ASIO_CONSTEXPR size_type size() const ASIO_NOEXCEPT {return length(); }
  ASIO_CONSTEXPR size_type length() const ASIO_NOEXCEPT {return traits_type::length(view_); }
  ASIO_CONSTEXPR size_type max_size() const ASIO_NOEXCEPT {return std::numeric_limits<int64_t>::max() / sizeof(CharT); }
  ASIO_NODISCARD ASIO_CONSTEXPR bool empty() const ASIO_NOEXCEPT {return *view_ == *null_char_(CharT{}); }

  ASIO_CONSTEXPR const_reference operator[](size_type pos) const  {return view_[pos] ;}
  ASIO_CONSTEXPR const_reference at(size_type pos) const
  {
    if (pos >= size())
      throw std::out_of_range("cstring-view out of range");
    return view_[pos];
  }
  ASIO_CONSTEXPR const_reference front() const  {return *view_;}
  ASIO_CONSTEXPR const_reference back()  const  {return view_[length() - 1];}
  ASIO_CONSTEXPR const_pointer data()    const ASIO_NOEXCEPT  {return view_;}
  ASIO_CONSTEXPR void remove_prefix(size_type n)  {view_ = view_ + n;}
  ASIO_CONSTEXPR void swap(basic_cstring_view& s) ASIO_NOEXCEPT  {std::swap(view_, s.view_);}

  ASIO_CONSTEXPR size_type copy(value_type* s, size_type n, size_type pos = 0) const
  {
    return traits_type::copy(s, view_ + pos, n) - view_;
  }
  ASIO_CONSTEXPR basic_cstring_view substr(size_type pos = 0) const
  {
    return basic_cstring_view(view_ + pos);
  }
  ASIO_CONSTEXPR string_view_type substr(size_type pos , size_type n) const {return string_view_type(view_+ pos, n);}

  ASIO_CONSTEXPR int compare(string_view_type s) const ASIO_NOEXCEPT
  {
    return traits_type::compare(view_, s.data(), std::min(length(), s.length()));
  }
  ASIO_CONSTEXPR int compare(size_type pos1, size_type n1, string_view_type s) const
  {
    return traits_type::compare(view_ + pos1, s.data(), std::min(n1, s.length()));
  }

  ASIO_CONSTEXPR int compare(size_type pos1, size_type n1, string_view_type s, size_type pos2, size_type n2) const
  {
    return traits_type::compare(view_ + pos1, s.data() + pos2, std::min(n1, n2));
  }

#if (__cplusplus >= 202002)
  ASIO_CONSTEXPR bool starts_with(string_view_type x) const ASIO_NOEXCEPT
  {
    return std::equal(view_, view_ + x.size(), x.begin(), x.end(), &traits_type::eq);
  }
  ASIO_CONSTEXPR bool starts_with(value_type x)       const ASIO_NOEXCEPT
  {
    return traits_type::eq(view_[0], x);
  }
  ASIO_CONSTEXPR bool ends_with(string_view_type x)   const ASIO_NOEXCEPT
  {
    return std::equal(view_ + x.size() - length(), view_ + length(), x.begin(), x.end(), &traits_type::eq);
  }
  ASIO_CONSTEXPR bool ends_with(value_type x)         const ASIO_NOEXCEPT
  {
    return !empty() && traits_type::eq(view_[length() - 1], x);
  }
#endif

  ASIO_CONSTEXPR size_type find(basic_cstring_view s, size_type pos = 0) const ASIO_NOEXCEPT
  {
    const auto e = end();
    const auto itr = std::search(begin(), e, s.begin(), s.end(), &traits_type::eq);
    return (itr != e) ? (itr - begin()) : npos;
  }
  ASIO_CONSTEXPR size_type find(value_type c, size_type pos = 0) const ASIO_NOEXCEPT
  {
    const auto e = end();
    const auto itr = std::find_if(begin(), e, [c](value_type cc) {return traits_type::eq(c, cc);});
    return (itr != e) ? (itr - begin()) : npos;
  }

  ASIO_CONSTEXPR size_type rfind(basic_cstring_view s, size_type pos = npos) const ASIO_NOEXCEPT
  {
    const auto b = rbegin();
    const auto itr = std::search(b, rend(), s.rbegin(), s.rend(), &traits_type::eq);
    return (itr != rend()) ? (itr - b) : npos;
  }
  ASIO_CONSTEXPR size_type rfind(value_type c, size_type pos = npos) const ASIO_NOEXCEPT
  {
    const auto b = rbegin();
    const auto itr = std::find_if(b, rend(), [c](value_type cc) {return traits_type::eq(c, cc);});
    return (itr != rend()) ? (itr - b) : npos;
  }


  friend ASIO_CONSTEXPR bool operator==(basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.compare(y) == 0;}
  friend ASIO_CONSTEXPR bool operator!=(basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.compare(y) != 0;}
  friend ASIO_CONSTEXPR bool operator< (basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.compare(y) <  0;}
  friend ASIO_CONSTEXPR bool operator> (basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.compare(y) >  0;}
  friend ASIO_CONSTEXPR bool operator<=(basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.compare(y) <= 0;}
  friend ASIO_CONSTEXPR bool operator>=(basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.compare(y) >= 0;}

 private:
  friend struct std::hash<basic_cstring_view>;
  ASIO_CONSTEXPR static const_pointer   null_char_()         {return null_char_(CharT{});}
  ASIO_CONSTEXPR static const char*     null_char_(char)     {return "\0";}
  ASIO_CONSTEXPR static const wchar_t*  null_char_(wchar_t)  {return L"\0";}
  ASIO_CONSTEXPR static const char16_t* null_char_(char16_t) {return u"\0";}
  ASIO_CONSTEXPR static const char32_t* null_char_(char32_t) {return U"\0";}
#if ASIO_HAS_CHAR8_T
  ASIO_CONSTEXPR static const char8_t* null_char_(char8_t) {return u8"\0";}
#endif

  const_pointer view_;
};

using cstring_view    = basic_cstring_view<char>;
using wcstring_view   = basic_cstring_view<wchar_t>;
using u16cstring_view = basic_cstring_view<char16_t>;
using u32cstring_view = basic_cstring_view<char32_t>;

#if ASIO_HAS_CHAR8_T
using u8cstring_view  = basic_cstring_view<char8_t>;
#endif

template struct basic_cstring_view<char>;
template struct basic_cstring_view<wchar_t>;
template struct basic_cstring_view<char16_t>;
template struct basic_cstring_view<char32_t>;

}

#if defined(ASIO_HAS_STD_HASH)

namespace std
{

template<typename CharT, typename Traits>
struct hash<asio::basic_cstring_view<CharT, Traits>>
{
  ASIO_CONSTEXPR std::size_t operator()(asio::basic_cstring_view<CharT, Traits> in) const
  {
    return std::hash<typename asio::basic_cstring_view<CharT, Traits>::string_view_type>()(in);
  }
};

}

# define ASIO_CSTRING_VIEW asio::cstring_view
# define ASIO_BASIC_CSTRING_VIEW(...) asio::basic_cstring_view < __VA_ARGS__ >
# define ASIO_CSTRING_VIEW_PARAM asio::cstring_view
# define ASIO_BASIC_CSTRING_VIEW_PARAM(...) asio::basic_cstring_view < __VA_ARGS__ >
#else // defined(ASIO_HAS_STRING_VIEW)
# define ASIO_CSTRING_VIEW std::string
# define ASIO_BASIC_CSTRING_VIEW(...) std::basic_string < __VA_ARGS __ >
# define ASIO_CSTRING_VIEW_PARAM const std::string&
# define ASIO_BASIC_CSTRING_VIEW_PARAM(...) const asio::cbasic_string < __VA_ARGS__ > &
#endif // defined(ASIO_HAS_STRING_VIEW)

#endif

#include "asio/detail/pop_options.hpp"

#endif //ASIO_CSTRING_VIEW_HPP
