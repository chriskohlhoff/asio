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

  using pointer                = typename string_view_type::pointer;
  using const_pointer          = typename string_view_type::const_pointer;
  using reference              = typename string_view_type::reference;
  using const_reference        = typename string_view_type::const_reference;
  using const_iterator         = typename string_view_type::const_iterator;
  using iterator               = typename string_view_type::iterator;
  using const_reverse_iterator = typename string_view_type::const_reverse_iterator;
  using reverse_iterator       = typename string_view_type::reverse_iterator;
  using size_type              = typename string_view_type::size_type;
  using difference_type        = typename string_view_type::difference_type;
  static ASIO_CONSTEXPR size_type npos = string_view_type::npos;

  ASIO_CONSTEXPR const_iterator begin() const ASIO_NOEXCEPT {return view_.begin();};
  ASIO_CONSTEXPR const_iterator end() const ASIO_NOEXCEPT {return view_.end();};
  ASIO_CONSTEXPR const_iterator cbegin() const ASIO_NOEXCEPT {return view_.cbegin();};
  ASIO_CONSTEXPR const_iterator cend() const ASIO_NOEXCEPT {return view_.cend();};
  ASIO_CONSTEXPR const_reverse_iterator rbegin() const ASIO_NOEXCEPT {return view_.rbegin();};
  ASIO_CONSTEXPR const_reverse_iterator rend() const ASIO_NOEXCEPT {return view_.rend();};
  ASIO_CONSTEXPR const_reverse_iterator crbegin() const ASIO_NOEXCEPT {return view_.crbegin();};
  ASIO_CONSTEXPR const_reverse_iterator crend() const ASIO_NOEXCEPT {return view_.crend();};

  ASIO_CONSTEXPR size_type size() const ASIO_NOEXCEPT {return view_.size(); }
  ASIO_CONSTEXPR size_type length() const ASIO_NOEXCEPT {return view_.length(); }
  ASIO_CONSTEXPR size_type max_size() const ASIO_NOEXCEPT {return view_.max_size(); }
  ASIO_NODISCARD ASIO_CONSTEXPR bool empty() const ASIO_NOEXCEPT {return view_.empty(); }

  ASIO_CONSTEXPR const_reference operator[](size_type pos) const  {return view_[pos] ;}
  ASIO_CONSTEXPR const_reference at(size_type pos) const  {return view_.at(pos);}
  ASIO_CONSTEXPR const_reference front() const  {return view_.front();}
  ASIO_CONSTEXPR const_reference back() const  {return view_.back();}
  ASIO_CONSTEXPR const_pointer data() const ASIO_NOEXCEPT  {return view_.data();}
  ASIO_CONSTEXPR void remove_prefix(size_type n)  {return view_.remove_prefix(n);}
  ASIO_CONSTEXPR void swap(basic_cstring_view& s) ASIO_NOEXCEPT  {view_.swap(s.view_);}

  ASIO_CONSTEXPR size_type copy(value_type* s, size_type n, size_type pos = 0) const {return view_.copy(s, n, pos);}
  ASIO_CONSTEXPR basic_cstring_view substr(size_type pos = 0) const
  {
    basic_cstring_view res;
    res.view_ = view_.substr(pos);
    return res;
  }
  ASIO_CONSTEXPR string_view_type substr(size_type pos , size_type n) const {return view_.substr(pos, n);}
  ASIO_CONSTEXPR int compare(string_view_type s) const ASIO_NOEXCEPT {return view_.compare(s);}
  ASIO_CONSTEXPR int compare(size_type pos1, size_type n1, string_view_type s) const {return view_.compare(pos1, n1, s); }
  ASIO_CONSTEXPR int compare(size_type pos1, size_type n1, string_view_type s, size_type pos2, size_type n2) const
  {
    return view_.compare(pos1, n1, s, pos2, n2);
  }
  ASIO_CONSTEXPR int compare(const value_type* s) const
  {
    return view_.compare(s);
  }
  ASIO_CONSTEXPR int compare(size_type pos1, size_type n1, const value_type* s) const
  {
    return view_.compare(pos1, n1, s);
  }
  ASIO_CONSTEXPR int compare(size_type pos1, size_type n1, const value_type* s, size_type n2) const
  {
    return view_.compare(pos1, n1, s, n2);
  }

#if (__cplusplus >= 202002)
  ASIO_CONSTEXPR bool starts_with(string_view_type x) const ASIO_NOEXCEPT {return view_.starts_with(x);}
  ASIO_CONSTEXPR bool starts_with(value_type x) const ASIO_NOEXCEPT {return view_.starts_with(x);}
  ASIO_CONSTEXPR bool starts_with(const value_type* x) const {return view_.starts_with(x);}
  ASIO_CONSTEXPR bool ends_with(string_view_type x) const ASIO_NOEXCEPT {return view_.ends_with(x);}
  ASIO_CONSTEXPR bool ends_with(value_type x) const ASIO_NOEXCEPT {return view_.ends_with(x);}
  ASIO_CONSTEXPR bool ends_with(const value_type* x) const {return view_.ends_with(x);}
#endif

  ASIO_CONSTEXPR size_type find(string_view_type s, size_type pos = 0) const ASIO_NOEXCEPT  {return view_.find(s, pos);}
  ASIO_CONSTEXPR size_type find(value_type c, size_type pos = 0) const ASIO_NOEXCEPT        {return view_.find(c, pos);}
  ASIO_CONSTEXPR size_type find(const value_type* s, size_type pos, size_type n) const {return view_.find(s, pos, n);}
  ASIO_CONSTEXPR size_type find(const value_type* s, size_type pos = 0) const          {return view_.find(s, pos);}
  ASIO_CONSTEXPR size_type rfind(string_view_type s, size_type pos = npos) const ASIO_NOEXCEPT {return view_.rfind(s, pos);}
  ASIO_CONSTEXPR size_type rfind(value_type c, size_type pos = npos) const ASIO_NOEXCEPT       {return view_.rfind(c, pos);}
  ASIO_CONSTEXPR size_type rfind(const value_type* s, size_type pos, size_type n) const   {return view_.rfind(s, pos, n);}
  ASIO_CONSTEXPR size_type rfind(const value_type* s, size_type pos = npos) const         {return view_.rfind(s, pos);}
  ASIO_CONSTEXPR size_type find_first_of(string_view_type s, size_type pos = 0) const ASIO_NOEXCEPT  {return view_.find_first_of(s, pos);}
  ASIO_CONSTEXPR size_type find_first_of(value_type c, size_type pos = 0) const ASIO_NOEXCEPT        {return view_.find_first_of(c, pos);}
  ASIO_CONSTEXPR size_type find_first_of(const value_type* s, size_type pos, size_type n) const {return view_.find_first_of(s, pos, n);}
  ASIO_CONSTEXPR size_type find_first_of(const value_type* s, size_type pos = 0) const          {return view_.find_first_of(s, pos);}
  ASIO_CONSTEXPR size_type find_last_of(string_view_type s, size_type pos = npos) const ASIO_NOEXCEPT  {return view_.find_last_of(s, pos);}
  ASIO_CONSTEXPR size_type find_last_of(value_type c, size_type pos = npos) const ASIO_NOEXCEPT        {return view_.find_last_of(c, pos);}
  ASIO_CONSTEXPR size_type find_last_of(const value_type* s, size_type pos, size_type n) const    {return view_.find_last_of(s, pos, n);}
  ASIO_CONSTEXPR size_type find_last_of(const value_type* s, size_type pos = npos) const          {return view_.find_last_of(s, pos);}
  ASIO_CONSTEXPR size_type find_first_not_of(string_view_type s, size_type pos = 0) const ASIO_NOEXCEPT  {return view_.find_first_not_of(s, pos);}
  ASIO_CONSTEXPR size_type find_first_not_of(value_type c, size_type pos = 0) const ASIO_NOEXCEPT        {return view_.find_first_not_of(c, pos);}
  ASIO_CONSTEXPR size_type find_first_not_of(const value_type* s, size_type pos, size_type n) const {return view_.find_first_not_of(s, pos, n);}
  ASIO_CONSTEXPR size_type find_first_not_of(const value_type* s, size_type pos = 0) const          {return view_.find_first_not_of(s, pos);}
  ASIO_CONSTEXPR size_type find_last_not_of(string_view_type s, size_type pos = npos) const ASIO_NOEXCEPT {return view_.find_last_not_of(s, pos);}
  ASIO_CONSTEXPR size_type find_last_not_of(value_type c, size_type pos = npos) const ASIO_NOEXCEPT       {return view_.find_last_not_of(c, pos);}
  ASIO_CONSTEXPR size_type find_last_not_of(const value_type* s, size_type pos, size_type n) const   {return view_.find_last_not_of(s, pos, n);}
  ASIO_CONSTEXPR size_type find_last_not_of(const value_type* s, size_type pos = npos) const         {return view_.find_last_not_of(s, pos);}

  friend ASIO_CONSTEXPR bool operator==(basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.view_ == y.view_;}
  friend ASIO_CONSTEXPR bool operator!=(basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.view_ != y.view_;}
  friend ASIO_CONSTEXPR bool operator< (basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.view_ <  y.view_;}
  friend ASIO_CONSTEXPR bool operator> (basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.view_ >  y.view_;}
  friend ASIO_CONSTEXPR bool operator<=(basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.view_ <= y.view_;}
  friend ASIO_CONSTEXPR bool operator>=(basic_cstring_view x, basic_cstring_view y) ASIO_NOEXCEPT {return x.view_ >= y.view_;}

 private:
  ASIO_CONSTEXPR static const char*     null_char_(char)     {return "\0";}
  ASIO_CONSTEXPR static const wchar_t*  null_char_(wchar_t)  {return L"\0";}
  ASIO_CONSTEXPR static const char16_t* null_char_(char16_t) {return u"\0";}
  ASIO_CONSTEXPR static const char32_t* null_char_(char32_t) {return U"\0";}
#if ASIO_HAS_CHAR8_T
  ASIO_CONSTEXPR static const char8_t* null_char_(char8_t) {return u8"\0";}
#endif

  basic_string_view<value_type, Traits> view_;
};

using cstring_view    = basic_cstring_view<char>;
using wcstring_view   = basic_cstring_view<wchar_t>;
using u16cstring_view = basic_cstring_view<char16_t>;
using u32cstring_view = basic_cstring_view<char32_t>;

#if ASIO_HAS_CHAR8_T
using u8cstring_view  = basic_cstring_view<char8_t>;
#endif

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
