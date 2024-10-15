//
// process/environment.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef ASIO_ENVIRONMENT_HPP
#define ASIO_ENVIRONMENT_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)


#include "asio/detail/config.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/codecvt.hpp"

#if !defined(GENERATING_DOCUMENTATION)
#if defined(ASIO_WINDOWS)
#include "asio/detail/environment_win.hpp"
#else
#include "asio/detail/environment_posix.hpp"
#endif
#endif

#include "asio/detail/codecvt.hpp"
#include "asio/detail/filesystem.hpp"
#include "asio/detail/quoted.hpp"
#include "asio/detail/type_traits.hpp"

#include <compare>
#include <iomanip>
#include <locale>
#include <string_view>
#include <optional>

#if ASIO_HAS_FILESYSTEM
#include <filesystem>
#endif

#include "asio/detail/push_options.hpp"


namespace asio
{

/// Namespace for information and functions regarding the calling process.
namespace environment
{

#if defined(GENERATING_DOCUMENTATION)

/// A char traits type that reflects the OS rules for string representing environment keys.
/** Can be an alias of std::char_traits. May only be defined for `char` and `wchar_t`.
*/
tempalte<typename Char>
using key_char_traits = implementation-defined ;

/// A char traits type that reflects the OS rules for string representing environment values.
/** Can be an alias of std::char_traits. May only be defined for `char` and `wchar_t`.
*/
tempalte<typename Char>
using value_char_traits = implementation-defined ;

/// The character type used by the environment. Either `char` or `wchar_t`.
using char_type = implementation-defined ;

/// The equal character in an environment string used to separate key and value.
constexpr char_type equality_sign = implementation-defined;

/// The delimiter in environemtn lists. Commonly used by the `PATH` variable.
constexpr char_type equality_sign = implementation-defined;

/// The native handle of an environment. Note that this can be an owning pointer and is generally not thread safe.
using native_handle = implementation-defined;

#endif


/// The iterator used by a value or value_view to iterator through environments that are lists.
struct value_iterator
{
    using string_view_type  = typename decay<ASIO_BASIC_STRING_VIEW_PARAM(char_type)>::type;
    using difference_type   = std::size_t;
    using value_type        = string_view_type;
    using pointer           = const string_view_type *;
    using reference         = const string_view_type & ;
    using iterator_category = std::forward_iterator_tag;

    value_iterator & operator++()
    {
        const auto delim = view_.find(delimiter);
        if (delim != string_view_type::npos)
          view_ = view_.substr(delim + 1);
        else
          view_ = view_.substr(view_.size());
        return *this;
    }

    value_iterator operator++(int)
    {
        auto last = *this;
        ++(*this);
        return last;
    }
    string_view_type operator*() const
    {
      const auto delim = view_.find(delimiter);
      if (delim == string_view_type::npos)
        return view_;
      else
        return view_.substr(0, delim);
    }
    const std::optional<string_view_type> operator->() const
    {
        return **this;
    }


    value_iterator() = default;
    value_iterator(const value_iterator & ) = default;
    value_iterator(ASIO_BASIC_CSTRING_VIEW_PARAM(char_type, value_char_traits<char_type>) view,
                   std::size_t offset = 0u) : view_(view.substr(offset))
    {
    }

#if defined(ASIO_HAS_STRING_VIEW)
    friend bool operator==(const value_iterator & l, const value_iterator & r) { return l.view_.data() == r.view_.data(); }
    friend bool operator!=(const value_iterator & l, const value_iterator & r) { return l.view_.data() != r.view_.data(); }
    friend bool operator<=(const value_iterator & l, const value_iterator & r) { return l.view_.data() <= r.view_.data(); }
    friend bool operator>=(const value_iterator & l, const value_iterator & r) { return l.view_.data() >= r.view_.data(); }
    friend bool operator< (const value_iterator & l, const value_iterator & r) { return l.view_.data() <  r.view_.data(); }
    friend bool operator> (const value_iterator & l, const value_iterator & r) { return l.view_.data() >  r.view_.data(); }
#else
    friend bool operator==(const value_iterator & l, const value_iterator & r) { return l.view_ == r.view_; }
    friend bool operator!=(const value_iterator & l, const value_iterator & r) { return l.view_ != r.view_; }
    friend bool operator<=(const value_iterator & l, const value_iterator & r) { return l.view_ <= r.view_; }
    friend bool operator>=(const value_iterator & l, const value_iterator & r) { return l.view_ >= r.view_; }
    friend bool operator< (const value_iterator & l, const value_iterator & r) { return l.view_ <  r.view_; }
    friend bool operator> (const value_iterator & l, const value_iterator & r) { return l.view_ >  r.view_; }
#endif
  private:
    string_view_type view_;
};

/// A view type for a key of an environment
struct key_view
{
    using value_type       = char_type;
    using traits_type      = key_char_traits<char_type>;
    using string_view_type = typename decay<ASIO_BASIC_STRING_VIEW_PARAM(char_type, traits_type)>::type;
    using string_type      = std::basic_string<char_type, key_char_traits<char_type>>;

    key_view() noexcept = default;
    key_view( const key_view& p ) = default;
    key_view( key_view&& p ) noexcept = default;
    template<typename Source, typename = typename constraint<is_constructible<string_view_type, Source>::value>::type>
    key_view( const Source& source ) : value_(source) {}
    key_view( const char_type * p) : value_(p) {}
    key_view(       char_type * p) : value_(p) {}

    ~key_view() = default;

    key_view& operator=( const key_view& p ) = default;
    key_view& operator=( key_view&& p ) noexcept = default;
    key_view& operator=( string_view_type source )
    {
        value_ = source;
        return *this;
    }

    void swap( key_view& other ) noexcept
    {
        std::swap(value_, other.value_);
    }

    string_view_type native() const noexcept {return value_;}

    operator string_view_type() const {return native();}

    int compare( const key_view& p ) const noexcept {return value_.compare(p.value_);}
    int compare( string_view_type str ) const {return value_.compare(str);}
    int compare( const value_type* s ) const {return value_.compare(s);}

    template< class CharT, class Traits = std::char_traits<CharT>,
            class Alloc = std::allocator<CharT> >
    std::basic_string<CharT,Traits,Alloc>
    string( const Alloc& alloc = Alloc(), const std::locale & loc = std::locale()) const
    {
        return asio::detail::convert_chars<Traits>(value_.data(), value_.data() + value_.size(), CharT(), alloc, loc);
    }

    std::string       string() const {return string<char>();}
    std::wstring     wstring() const {return string<wchar_t>();}
    std::u16string u16string() const {return string<char16_t>();}
    std::u32string u32string() const {return string<char32_t>();}

    string_type native_string() const
    {
        return string<char_type, key_char_traits<char_type>>();
    }

#if ASIO_HAS_CHAR8_T
    std::u8string   u8string() const {return string<char8_t>();}
#endif

    bool empty() const {return value_.empty(); }

    friend bool operator==(key_view l, key_view r) { return l.value_ == r.value_; }
    friend bool operator!=(key_view l, key_view r) { return l.value_ != r.value_; }
    friend bool operator<=(key_view l, key_view r) { return l.value_ <= r.value_; }
    friend bool operator>=(key_view l, key_view r) { return l.value_ >= r.value_; }
    friend bool operator< (key_view l, key_view r) { return l.value_ <  r.value_; }
    friend bool operator> (key_view l, key_view r) { return l.value_ >  r.value_; }

    template< class CharT, class Traits >
    friend std::basic_ostream<CharT,Traits>&
    operator<<( std::basic_ostream<CharT,Traits>& os, const key_view& p )
    {
        os << asio::quoted(p.string<CharT,Traits>());
        return os;
    }

    template< class CharT, class Traits >
    friend std::basic_istream<CharT,Traits>&
    operator>>( std::basic_istream<CharT,Traits>& is, key_view& p )
    {
        std::basic_string<CharT, Traits> t;
        is >> asio::quoted(t);
        p = t;
        return is;
    }
    const value_type * data() const {return value_.data(); }
    std::size_t size() const {return value_.size(); }
  private:
    string_view_type value_;
};


struct value_view
{
    using value_type       = char_type;
    using string_view_type = ASIO_BASIC_CSTRING_VIEW(char_type, value_char_traits<char_type>);
    using string_type      = std::basic_string<char_type, value_char_traits<char_type>>;
    using traits_type      = value_char_traits<char_type>;

    value_view() noexcept = default;
    value_view( const value_view& p ) = default;
    value_view( value_view&& p ) noexcept = default;
    template<typename Source, typename = typename constraint<is_constructible<string_view_type, Source>::value>::type>
    value_view( const Source& source ) : value_(source) {}
    value_view( const char_type * p) : value_(p) {}
    value_view(       char_type * p) : value_(p) {}

    ~value_view() = default;

    value_view& operator=( const value_view& p ) = default;
    value_view& operator=( value_view&& p ) noexcept = default;
    value_view& operator=( string_view_type source )
    {
        value_ = source;
        return *this;
    }

    void swap( value_view& other ) noexcept
    {
        std::swap(value_, other.value_);
    }

    string_view_type native() const noexcept {return value_;}

    operator string_view_type() const {return native();}
    operator typename string_view_type::string_view_type() const {return value_; }

    int compare( const value_view& p ) const noexcept {return value_.compare(p.value_);}
    int compare( string_view_type str ) const {return value_.compare(str);}
    int compare( const value_type* s ) const {return value_.compare(s);}

    template< class CharT, class Traits = std::char_traits<CharT>,
            class Alloc = std::allocator<CharT> >
    std::basic_string<CharT,Traits,Alloc>
    string( const Alloc& alloc = Alloc(), const std::locale & loc = std::locale() ) const
    {
        return asio::detail::convert_chars<Traits>(value_.begin(), value_.end(), CharT(), alloc, loc);
    }

    std::string string() const       {return string<char>();}
    std::wstring wstring() const     {return string<wchar_t>();}
    std::u16string u16string() const {return string<char16_t>();}
    std::u32string u32string() const {return string<char32_t>();}

    string_type native_string() const
    {
        return string<char_type, value_char_traits<char_type>>();
    }

#if ASIO_HAS_CHAR8_T
    std::u8string u8string() const   {return string<char8_t>();}
#endif

    bool empty() const {return value_.empty(); }

    friend bool operator==(value_view l, value_view r) { return l.value_ == r.value_; }
    friend bool operator!=(value_view l, value_view r) { return l.value_ != r.value_; }
    friend bool operator<=(value_view l, value_view r) { return l.value_ <= r.value_; }
    friend bool operator>=(value_view l, value_view r) { return l.value_ >= r.value_; }
    friend bool operator< (value_view l, value_view r) { return l.value_ <  r.value_; }
    friend bool operator> (value_view l, value_view r) { return l.value_ >  r.value_; }

    template< class CharT, class Traits >
    friend std::basic_ostream<CharT,Traits>&
    operator<<( std::basic_ostream<CharT,Traits>& os, const value_view& p )
    {
        os << asio::quoted(p.string<CharT,Traits>());
        return os;
    }

    template< class CharT, class Traits >
    friend std::basic_istream<CharT,Traits>&
    operator>>( std::basic_istream<CharT,Traits>& is, value_view& p )
    {
        std::basic_string<CharT, Traits> t;
        is >> asio::quoted(t);
        p = t;
        return is;
    }
    value_iterator begin() const {return value_iterator(value_.data());}
    value_iterator   end() const {return value_iterator(value_.data() , value_.size());}

    const char_type * c_str() {return value_.c_str(); }
    const value_type * data() const {return value_.data(); }
    std::size_t size() const {return value_.size(); }

  private:
    string_view_type value_;
};


struct key_value_pair_view
{
  using value_type       = char_type;
  using string_type      = std::basic_string<char_type>;
  using string_view_type = ASIO_BASIC_CSTRING_VIEW(char_type);
  using traits_type      = std::char_traits<char_type>;

  key_value_pair_view() noexcept = default;
  key_value_pair_view( const key_value_pair_view& p ) = default;
  key_value_pair_view( key_value_pair_view&& p ) noexcept = default;
  template<typename Source, typename = typename constraint<is_constructible<string_view_type, Source>::value>::type>
  key_value_pair_view( const Source& source ) : value_(source) {}

  key_value_pair_view( const char_type * p) : value_(p) {}
  key_value_pair_view(       char_type * p) : value_(p) {}


  ~key_value_pair_view() = default;

  key_value_pair_view& operator=( const key_value_pair_view& p ) = default;
  key_value_pair_view& operator=( key_value_pair_view&& p ) noexcept = default;

  void swap( key_value_pair_view& other ) noexcept
  {
      std::swap(value_, other.value_);
  }

  string_view_type native() const noexcept {return value_;}

  operator string_view_type() const {return native();}
  operator typename string_view_type::string_view_type() const {return value_; }

  int compare( const key_value_pair_view& p ) const noexcept {return value_.compare(p.value_);}
  int compare( const string_type& str ) const {return value_.compare(str);}
  int compare( string_view_type str ) const {return value_.compare(str);}
  int compare( const value_type* s ) const {return value_.compare(s);}

  template< class CharT, class Traits = std::char_traits<CharT>, class Alloc = std::allocator<CharT> >
  std::basic_string<CharT,Traits,Alloc>
  string( const Alloc& alloc = Alloc(), const std::locale & loc = std::locale()) const
  {
      return asio::detail::convert_chars<Traits>(value_.begin(), value_.end(), CharT(), alloc, loc);
  }

  std::string string() const       {return string<char>();}
  std::wstring wstring() const     {return string<wchar_t>();}
  std::u16string u16string() const {return string<char16_t>();}
  std::u32string u32string() const {return string<char32_t>();}

  string_type native_string() const
  {
    return string<char_type>();
  }

#if ASIO_HAS_CHAR8_T
  std::u8string u8string() const   {return string<char8_t>();}
#endif

  bool empty() const {return value_.empty(); }

  key_view key_view() const
  {
      const auto eq = value_.find(equality_sign);
      const auto res = native().substr(0,  eq == string_view_type::npos ? value_.size() : eq);
      return key_view::string_view_type(res.data(), res.size());
  }
  value_view value_view() const
  {
      return environment::value_view(native().substr(value_.find(equality_sign)  + 1));
  }

  friend bool operator==(const key_value_pair_view & l, const key_value_pair_view & r) { return l.value_ == r.value_; }
  friend bool operator!=(const key_value_pair_view & l, const key_value_pair_view & r) { return l.value_ != r.value_; }
  friend bool operator<=(const key_value_pair_view & l, const key_value_pair_view & r) { return l.value_ <= r.value_; }
  friend bool operator>=(const key_value_pair_view & l, const key_value_pair_view & r) { return l.value_ >= r.value_; }
  friend bool operator< (const key_value_pair_view & l, const key_value_pair_view & r) { return l.value_ <  r.value_; }
  friend bool operator> (const key_value_pair_view & l, const key_value_pair_view & r) { return l.value_ >  r.value_; }

  template< class CharT, class Traits >
  friend std::basic_ostream<CharT,Traits>&
  operator<<( std::basic_ostream<CharT,Traits>& os, const key_value_pair_view& p )
  {
      os << asio::quoted(p.string<CharT,Traits>());
      return os;
  }

  template< class CharT, class Traits >
  friend std::basic_istream<CharT,Traits>&
  operator>>( std::basic_istream<CharT,Traits>& is, key_value_pair_view& p )
  {
      std::basic_string<CharT, Traits> t;
      is >> asio::quoted(t);
      p = t;
      return is;
  }

  template<std::size_t Idx>
  inline auto get() const -> typename conditional<Idx == 0u, asio::environment::key_view,
                                                             asio::environment::value_view>::type;
  const value_type * c_str() const noexcept
  {
    return value_.data();
  }
  const value_type * data() const {return value_.data(); }
  std::size_t size() const {return value_.size(); }

 private:

  string_view_type value_;
};

template<>
key_view key_value_pair_view::get<0u>() const
{
    return key_view();
}

template<>
value_view key_value_pair_view::get<1u>() const
{
    return value_view();
}

struct key
{
    using value_type       = char_type;
    using traits_type      = key_char_traits<char_type>;
    using string_type      = std::basic_string<char_type, traits_type>;
    using string_view_type = typename decay<ASIO_BASIC_STRING_VIEW_PARAM(char_type, traits_type)>::type;

    key() noexcept = default;
    key( const key& p ) = default;
    key( key&& p ) noexcept = default;
    key( const string_type& source ) : value_(source) {}
    key( string_type&& source ) : value_(std::move(source)) {}
    key( const value_type * raw ) : value_(raw) {}
    key(       value_type * raw ) : value_(raw) {}

    key(key_view kv) : value_(kv) {}

    template< class Source >
    key( const Source& source, const std::locale& loc = std::locale(),
         decltype(source.data()) = nullptr)
        : value_(asio::detail::convert_chars<traits_type>(source.data(), source.data() + source.size(), char_type(), std::allocator<char_type>(), loc))
    {
    }

    key(const typename conditional<is_same<value_type, char>::value, wchar_t, char>::type  * raw, const std::locale& loc = std::locale())
        : value_(asio::detail::convert_chars<traits_type>(
                raw,
                raw + std::char_traits<asio::decay<asio::remove_pointer<decltype(raw)>::type>::type>::length(raw),
                char_type(), std::allocator<char_type>(), loc))
    {
    }

    key(const char16_t * raw, const std::locale& loc = std::locale()) : value_(asio::detail::convert_chars<traits_type>(raw,raw + std::char_traits<char16_t>::length(raw), char_type(), std::allocator<char_type>(), loc)) {}
    key(const char32_t * raw, const std::locale& loc = std::locale()) : value_(asio::detail::convert_chars<traits_type>(raw,raw + std::char_traits<char32_t>::length(raw), char_type(), std::allocator<char_type>(), loc)) {}
#if ASIO_HAS_CHAR8_T
    key(const char8_t * raw, const std::locale& loc = std::locale()) : value_(asio::detail::convert_chars<traits_type>(raw,raw + std::char_traits<char8_t>::length(raw), char_type(), std::allocator<char_type>(), loc)) {}
#endif

    template<typename Char, typename Traits>
    key(std::basic_string_view<Char, Traits> source, const std::locale& loc = std::locale())
        : value_(asio::detail::convert_chars<traits_type>(source.data(), source.data() + source.size(), char_type(), std::allocator<char_type>(), loc))
    {
    }

    template< class InputIt >
    key( InputIt first, InputIt last, const std::locale& loc = std::locale())
    : key(std::basic_string(first, last), loc)
    {
    }

    ~key() = default;

    key& operator=( const key& p ) = default;
    key& operator=( key&& p ) noexcept = default;
    key& operator=( string_type&& source )
    {
        value_ = std::move(source);
        return *this;
    }
    template< class Source >
    key& operator=( const Source& source )
    {
        value_ = asio::detail::convert_chars<traits_type>(source.data(), source.data() + source.size(), char_type(), std::allocator<char_type>());
        return *this;
    }

    key& assign( string_type&& source )
    {
        value_ = std::move(source);
        return *this;
    }
    template< class Source >
    key& assign( const Source& source , const std::locale & loc)
    {
        value_ = asio::detail::convert_chars<traits_type>(source.data(), source.data() + source.size(), char_type(), std::allocator<char_type>(), loc);
        return *this;
    }

    template< class InputIt >
    key& assign( InputIt first, InputIt last )
    {
        return assign(std::string(first, last));
    }

    void clear() {value_.clear();}

    void swap( key& other ) noexcept
    {
        std::swap(value_, other.value_);
    }

    const value_type* c_str() const noexcept {return value_.c_str();}
    const string_type& native() const noexcept {return value_;}
    string_view_type native_view() const noexcept {return value_;}

    operator string_type() const {return native();}
    operator string_view_type() const {return native_view();}

    int compare( const key& p ) const noexcept {return value_.compare(p.value_);}
    int compare( const string_type& str ) const {return value_.compare(str);}
    int compare( string_view_type str ) const {return value_.compare(str);}
    int compare( const value_type* s ) const {return value_.compare(s);}

    template< class CharT, class Traits = std::char_traits<CharT>,
            class Alloc = std::allocator<CharT> >
    std::basic_string<CharT,Traits,Alloc>
    string( const Alloc& alloc = Alloc(), const std::locale & loc = std::locale() ) const
    {
        return asio::detail::convert_chars<Traits>(value_.data(), value_.data() + value_.size(), CharT(), alloc, loc);
    }

    std::string string() const       {return string<char>();}
    std::wstring wstring() const     {return string<wchar_t>();}
    std::u16string u16string() const {return string<char16_t>();}
    std::u32string u32string() const {return string<char32_t>();}

    std::basic_string<char_type, value_char_traits<char_type>> native_string() const
    {
        return string<char_type, value_char_traits<char_type>>();
    }

#if ASIO_HAS_CHAR8_T
    std::u8string u8string() const   {return string<char8_t>();}
#endif

    bool empty() const {return value_.empty(); }

    friend bool operator==(const key & l, const key & r) { return l.value_ == r.value_; }
    friend bool operator!=(const key & l, const key & r) { return l.value_ != r.value_; }
    friend bool operator<=(const key & l, const key & r) { return l.value_ <= r.value_; }
    friend bool operator>=(const key & l, const key & r) { return l.value_ >= r.value_; }
    friend bool operator< (const key & l, const key & r) { return l.value_ <  r.value_; }
    friend bool operator> (const key & l, const key & r) { return l.value_ >  r.value_; }

    template< class CharT, class Traits >
    friend std::basic_ostream<CharT,Traits>&
    operator<<( std::basic_ostream<CharT,Traits>& os, const key& p )
    {
        os << asio::quoted(p.string<CharT,Traits>());
        return os;
    }

    template< class CharT, class Traits >
    friend std::basic_istream<CharT,Traits>&
    operator>>( std::basic_istream<CharT,Traits>& is, key& p )
    {
        std::basic_string<CharT, Traits> t;
        is >> asio::quoted(t);
        p = t;
        return is;
    }
    const value_type * data() const {return value_.data(); }
    std::size_t size() const {return value_.size(); }

  private:
    string_type value_;
};

template<typename T, typename =  typename constraint<!is_constructible<key_view, T>::value>::type> inline bool operator==(const key_view & l, const T & r) { return l == key_view(key(r)); }
template<typename T, typename =  typename constraint<!is_constructible<key_view, T>::value>::type> inline bool operator!=(const key_view & l, const T & r) { return l != key_view(key(r)); }
template<typename T, typename =  typename constraint<!is_constructible<key_view, T>::value>::type> inline bool operator<=(const key_view & l, const T & r) { return l <= key_view(key(r)); }
template<typename T, typename =  typename constraint<!is_constructible<key_view, T>::value>::type> inline bool operator>=(const key_view & l, const T & r) { return l >= key_view(key(r)); }
template<typename T, typename =  typename constraint<!is_constructible<key_view, T>::value>::type> inline bool operator< (const key_view & l, const T & r) { return l <  key_view(key(r)); }
template<typename T, typename =  typename constraint<!is_constructible<key_view, T>::value>::type> inline bool operator> (const key_view & l, const T & r) { return l >  key_view(key(r)); }

template<typename T> inline typename constraint<!is_constructible<key_view, T>::value, bool>::type operator==(const T & l, const key_view & r) { return key_view(key(l)) == r; }
template<typename T> inline typename constraint<!is_constructible<key_view, T>::value, bool>::type operator!=(const T & l, const key_view & r) { return key_view(key(l)) != r; }
template<typename T> inline typename constraint<!is_constructible<key_view, T>::value, bool>::type operator<=(const T & l, const key_view & r) { return key_view(key(l)) <= r; }
template<typename T> inline typename constraint<!is_constructible<key_view, T>::value, bool>::type operator>=(const T & l, const key_view & r) { return key_view(key(l)) >= r; }
template<typename T> inline typename constraint<!is_constructible<key_view, T>::value, bool>::type operator< (const T & l, const key_view & r) { return key_view(key(l)) <  r; }
template<typename T> inline typename constraint<!is_constructible<key_view, T>::value, bool>::type operator> (const T & l, const key_view & r) { return key_view(key(l)) >  r; }


struct value
{
    using value_type       = char_type;
    using traits_type      = value_char_traits<char_type>;
    using string_type      = std::basic_string<char_type, traits_type>;
    using string_view_type = ASIO_BASIC_CSTRING_VIEW(char_type, traits_type);

    value() noexcept = default;
    value( const value& p ) = default;

    value( const string_type& source ) : value_(source) {}
    value( string_type&& source ) : value_(std::move(source)) {}
    value( const value_type * raw ) : value_(raw) {}
    value(       value_type * raw ) : value_(raw) {}


    value(value_view kv) : value_(kv.c_str()) {}

    template< class Source >
    value( const Source& source, const std::locale& loc = std::locale(),
         decltype(source.data()) = nullptr)
            : value_(asio::detail::convert_chars<traits_type>(source.data(), source.data() + source.size(), char_type(), std::allocator<char_type>(), loc))
    {
    }

    value(const typename conditional<is_same<value_type, char>::value, wchar_t, char>::type  * raw, const std::locale& loc = std::locale())
            : value_(asio::detail::convert_chars<traits_type>(
            raw,
            raw + std::char_traits<asio::decay<asio::remove_pointer<decltype(raw)>::type>::type>::length(raw),
            char_type(), std::allocator<char_type>(), loc))
    {
    }

    value(const char16_t * raw, const std::locale& loc = std::locale()) : value_(asio::detail::convert_chars<traits_type>(raw,raw + std::char_traits<char16_t>::length(raw), char_type(), std::allocator<char_type>(), loc)) {}
    value(const char32_t * raw, const std::locale& loc = std::locale()) : value_(asio::detail::convert_chars<traits_type>(raw,raw + std::char_traits<char32_t>::length(raw), char_type(), std::allocator<char_type>(), loc)) {}
#if ASIO_HAS_CHAR8_T
    value(const char8_t * raw, const std::locale& loc = std::locale()) : value_(asio::detail::convert_chars<traits_type>(raw,raw + std::char_traits<char8_t>::length(raw), char_type(), std::allocator<char_type>(), loc)) {}
#endif

    template< class InputIt >
    value( InputIt first, InputIt last, const std::locale& loc = std::locale())
            : value(std::basic_string(first, last), loc)
    {
    }

    ~value() = default;

    value& operator=( const value& p ) = default;
    value& operator=( value&& p ) noexcept = default;
    value& operator=( string_type&& source )
    {
        value_ = std::move(source);
        return *this;
    }
    template< class Source >
    value& operator=( const Source& source )
    {
        value_ = asio::detail::convert_chars<traits_type>(source.data(), source.data() + source.size(), char_type(), std::allocator<char_type>());
        return *this;
    }

    value& assign( string_type&& source )
    {
        value_ = std::move(source);
        return *this;
    }
    template< class Source >
    value& assign( const Source& source, const std::locale & loc = std::locale() )
    {
        value_ = asio::detail::convert_chars<traits_type>(source.data(), source.data() + source.size(), char_type(), std::allocator<char_type>(), loc);
        return *this;
    }

    template< class InputIt >
    value& assign( InputIt first, InputIt last )
    {
        return assign(std::string(first, last));
    }

    void push_back(const value & sv)
    {
        (value_ += delimiter) += sv;
    }

    void clear() {value_.clear();}

    void swap( value& other ) noexcept
    {
        std::swap(value_, other.value_);
    }

    const value_type* c_str() const noexcept {return value_.c_str();}
    const string_type& native() const noexcept {return value_;}
    string_view_type native_view() const noexcept {return value_;}

    operator string_type() const {return native();}
    operator string_view_type() const {return native_view();}
    operator typename string_view_type::string_view_type() const {return value_; }

    int compare( const value& p ) const noexcept {return value_.compare(p.value_);}
    int compare( const string_type& str ) const {return value_.compare(str);}
    int compare( string_view_type str ) const {return value_.compare(str);}
    int compare( const value_type* s ) const {return value_.compare(s);}

    template< class CharT, class Traits = std::char_traits<CharT>,
            class Alloc = std::allocator<CharT> >
    std::basic_string<CharT,Traits,Alloc>
    string( const Alloc& alloc = Alloc(), const std::locale & loc = std::locale()) const
    {
        return asio::detail::convert_chars<Traits>(value_.data(), value_.data() + value_.size(), CharT(), alloc, loc);
    }

    std::string string() const       {return string<char>();}
    std::wstring wstring() const     {return string<wchar_t>();}
    std::u16string u16string() const {return string<char16_t>();}
    std::u32string u32string() const {return string<char32_t>();}

    std::basic_string<char_type, value_char_traits<char_type>> native_string() const
    {
        return string<char_type, value_char_traits<char_type>>();
    }

#if ASIO_HAS_CHAR8_T
    std::u8string u8string() const   {return string<char8_t>();}
#endif

    bool empty() const {return value_.empty(); }


    friend bool operator==(const value & l, const value & r) { return l.value_ == r.value_; }
    friend bool operator!=(const value & l, const value & r) { return l.value_ != r.value_; }
    friend bool operator<=(const value & l, const value & r) { return l.value_ <= r.value_; }
    friend bool operator>=(const value & l, const value & r) { return l.value_ >= r.value_; }
    friend bool operator< (const value & l, const value & r) { return l.value_ <  r.value_; }
    friend bool operator> (const value & l, const value & r) { return l.value_ >  r.value_; }

    template< class CharT, class Traits >
    friend std::basic_ostream<CharT,Traits>&
    operator<<( std::basic_ostream<CharT,Traits>& os, const value& p )
    {
        os << asio::quoted(p.string<CharT,Traits>());
        return os;
    }

    template< class CharT, class Traits >
    friend std::basic_istream<CharT,Traits>&
    operator>>( std::basic_istream<CharT,Traits>& is, value& p )
    {
        std::basic_string<CharT, Traits> t;
        is >> asio::quoted(t);
        p = t;
        return is;
    }

    value_iterator begin() const {return value_iterator(value_.data());}
    value_iterator   end() const {return value_iterator(value_.data(), value_.size());}
  const value_type * data() const {return value_.data(); }
  std::size_t size() const {return value_.size(); }

  private:
    string_type value_;
};



template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator==(const value_view & l, const T & r) { return l == value_view(value(r)); }
template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator!=(const value_view & l, const T & r) { return l != value_view(value(r)); }
template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator<=(const value_view & l, const T & r) { return l <= value_view(value(r)); }
template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator>=(const value_view & l, const T & r) { return l >= value_view(value(r)); }
template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator< (const value_view & l, const T & r) { return l <  value_view(value(r)); }
template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator> (const value_view & l, const T & r) { return l >  value_view(value(r)); }

template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator==(const T & l, const value_view & r) { return value_view(value(l)) == r; }
template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator!=(const T & l, const value_view & r) { return value_view(value(l)) != r; }
template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator<=(const T & l, const value_view & r) { return value_view(value(l)) <= r; }
template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator>=(const T & l, const value_view & r) { return value_view(value(l)) >= r; }
template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator< (const T & l, const value_view & r) { return value_view(value(l)) <  r; }
template<typename T, typename =  typename constraint<!is_constructible<value_view, T>::value>::type> inline bool operator> (const T & l, const value_view & r) { return value_view(value(l)) >  r; }


struct key_value_pair
{
    using value_type       = char_type;
    using traits_type      = std::char_traits<char_type>;
    using string_type      = std::basic_string<char_type>;
    using string_view_type = ASIO_BASIC_CSTRING_VIEW(char_type);

    key_value_pair() noexcept = default;
    key_value_pair( const key_value_pair& p ) = default;
    key_value_pair( key_value_pair&& p ) noexcept = default;
    key_value_pair(key_view key, value_view value) : value_(key.string<char_type>() + equality_sign + value.string<char_type>()) {}
    key_value_pair( const string_type& source ) : value_(source) {}
    key_value_pair( string_type&& source ) : value_(std::move(source)) {}
    key_value_pair( const value_type * raw ) : value_(raw) {}
    key_value_pair(       value_type * raw ) : value_(raw) {}


    key_value_pair(key_value_pair_view kv) : value_(kv.c_str()) {}

    template< class Source >
    key_value_pair( const Source& source, const std::locale& loc = std::locale(),
           decltype(source.data()) = nullptr)
            : value_(asio::detail::convert_chars<traits_type>(source.data(), source.data() + source.size(), char_type(), std::allocator<char_type>(), loc))
    {
    }

    key_value_pair(const typename conditional<is_same<value_type, char>::value, wchar_t, char>::type  * raw, const std::locale& loc = std::locale())
            : value_(asio::detail::convert_chars<traits_type>(
                     raw,
                     raw + std::char_traits<asio::decay<asio::remove_pointer<decltype(raw)>::type>::type>::length(raw),
                     char_type(), std::allocator<char_type>(), loc))
    {
    }

    key_value_pair(const char16_t * raw, const std::locale& loc = std::locale()) : value_(asio::detail::convert_chars<traits_type>(raw,raw + std::char_traits<char16_t>::length(raw), char_type(), std::allocator<char_type>(), loc)) {}
    key_value_pair(const char32_t * raw, const std::locale& loc = std::locale()) : value_(asio::detail::convert_chars<traits_type>(raw,raw + std::char_traits<char32_t>::length(raw), char_type(), std::allocator<char_type>(), loc)) {}
#if ASIO_HAS_CHAR8_T
            key_value_pair(const char8_t * raw, const std::locale& loc = std::locale()) : value_(asio::detail::convert_chars<traits_type>(raw,raw + std::char_traits<char8_t>::length(raw), char_type(), std::allocator<char_type>(), loc)) {}
#endif

    template< class InputIt >
    key_value_pair( InputIt first, InputIt last, const std::locale& loc = std::locale())
            : key_value_pair(std::basic_string(first, last), loc)
    {
    }

    ~key_value_pair() = default;

    key_value_pair& operator=( const key_value_pair& p ) = default;
    key_value_pair& operator=( key_value_pair&& p ) noexcept = default;
    key_value_pair& operator=( string_type&& source )
    {
        value_ = std::move(source);
        return *this;
    }
    template< class Source >
    key_value_pair& operator=( const Source& source )
    {
        value_ = asio::detail::convert_chars<traits_type>(source.data(), source.data() + source.size(), char_type(), std::allocator<char_type>());
        return *this;
    }

    key_value_pair& assign( string_type&& source )
    {
        value_ = std::move(source);
        return *this;
    }
    template< class Source >
    key_value_pair& assign( const Source& source, const std::locale & loc = std::locale() )
    {
        value_ = asio::detail::convert_chars<traits_type>(source.data(), source.data() + source.size(), char_type(), std::allocator<char_type>(), loc);
        return *this;
    }

    template< class InputIt >
    key_value_pair& assign( InputIt first, InputIt last )
    {
        return assign(std::string(first, last));
    }

    void clear() {value_.clear();}

    void swap( key_value_pair& other ) noexcept
    {
        std::swap(value_, other.value_);
    }

    const value_type* c_str() const noexcept {return value_.c_str();}
    const string_type& native() const noexcept {return value_;}
    string_view_type native_view() const noexcept {return value_;}

    operator string_type() const {return native();}
    operator string_view_type() const {return native_view();}

    int compare( const key_value_pair& p ) const noexcept {return value_.compare(p.value_);}
    int compare( const string_type& str ) const {return value_.compare(str);}
    int compare( string_view_type str ) const {return value_.compare(str);}
    int compare( const value_type* s ) const {return value_.compare(s);}

    template< class CharT, class Traits = std::char_traits<CharT>, class Alloc = std::allocator<CharT> >
    std::basic_string<CharT,Traits,Alloc>
    string( const Alloc& alloc = Alloc(), const std::locale & loc = std::locale() ) const
    {
        return asio::detail::convert_chars<Traits>(value_.data(), value_.data() + value_.size(), CharT(), alloc, loc);
    }

    std::string string() const       {return string<char>();}
    std::wstring wstring() const     {return string<wchar_t>();}
    std::u16string u16string() const {return string<char16_t>();}
    std::u32string u32string() const {return string<char32_t>();}

    std::basic_string<char_type, value_char_traits<char_type>> native_string() const
    {
        return string<char_type, value_char_traits<char_type>>();
    }

#if ASIO_HAS_CHAR8_T
    std::u8string u8string() const   {return string<char8_t>();}
#endif

    bool empty() const {return value_.empty(); }

    key     key() const {return value_.substr(0, value_.find(equality_sign));}
    value value() const {return value_.substr(value_.find(equality_sign) + 1, string_type::npos);}

    key_view     key_view() const
    {
        const auto k = native_view().substr(0, value_.find(equality_sign));
        return asio::environment::key_view::string_view_type (k.data(), k.size());
    }
    value_view value_view() const {return value_view::string_view_type(native_view().substr(value_.find(equality_sign)  + 1));}


    friend bool operator==(const key_value_pair & l, const key_value_pair & r) { return l.value_ == r.value_; }
    friend bool operator!=(const key_value_pair & l, const key_value_pair & r) { return l.value_ != r.value_; }
    friend bool operator<=(const key_value_pair & l, const key_value_pair & r) { return l.value_ <= r.value_; }
    friend bool operator>=(const key_value_pair & l, const key_value_pair & r) { return l.value_ >= r.value_; }
    friend bool operator< (const key_value_pair & l, const key_value_pair & r) { return l.value_ <  r.value_; }
    friend bool operator> (const key_value_pair & l, const key_value_pair & r) { return l.value_ >  r.value_; }


    template< class CharT, class Traits >
    friend std::basic_ostream<CharT,Traits>&
    operator<<( std::basic_ostream<CharT,Traits>& os, const key_value_pair& p )
    {
        os << asio::quoted(p.string<CharT,Traits>());
        return os;
    }

    template< class CharT, class Traits >
    friend std::basic_istream<CharT,Traits>&
    operator>>( std::basic_istream<CharT,Traits>& is, key_value_pair& p )
    {
        std::basic_string<CharT, Traits> t;
        is >> asio::quoted(t);
        p = t;
        return is;
    }

    template<std::size_t Idx>
    auto get() const
    {
        if constexpr (Idx == 0u)
            return key_view();
        else
            return value_view();
    }

    template<std::size_t Idx>
    inline auto get() const -> typename conditional<Idx == 0u, asio::environment::key_view,
            asio::environment::value_view>::type;

    const value_type * data() const {return value_.data(); }
    std::size_t size() const {return value_.size(); }

private:

    string_type value_;
};

template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator==(const key_value_pair_view & l, const T & r) { return l == key_value_pair_view(key_value_pair(r)); }
template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator!=(const key_value_pair_view & l, const T & r) { return l != key_value_pair_view(key_value_pair(r)); }
template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator<=(const key_value_pair_view & l, const T & r) { return l <= key_value_pair_view(key_value_pair(r)); }
template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator>=(const key_value_pair_view & l, const T & r) { return l >= key_value_pair_view(key_value_pair(r)); }
template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator< (const key_value_pair_view & l, const T & r) { return l <  key_value_pair_view(key_value_pair(r)); }
template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator> (const key_value_pair_view & l, const T & r) { return l >  key_value_pair_view(key_value_pair(r)); }

template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator==(const T & l, const key_value_pair_view & r) { return key_value_pair_view(key_value_pair(l)) == r; }
template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator!=(const T & l, const key_value_pair_view & r) { return key_value_pair_view(key_value_pair(l)) != r; }
template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator<=(const T & l, const key_value_pair_view & r) { return key_value_pair_view(key_value_pair(l)) <= r; }
template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator>=(const T & l, const key_value_pair_view & r) { return key_value_pair_view(key_value_pair(l)) >= r; }
template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator< (const T & l, const key_value_pair_view & r) { return key_value_pair_view(key_value_pair(l)) <  r; }
template<typename T, typename = typename constraint<!is_constructible<key_value_pair_view, T>::value>::type> inline bool operator> (const T & l, const key_value_pair_view & r) { return key_value_pair_view(key_value_pair(l)) >  r; }

template<>
key_view key_value_pair::get<0u>() const
{
    return key_view();
}

template<>
value_view key_value_pair::get<1u>() const
{
    return value_view();
}

struct view
{
    using native_handle_type = environment::native_handle_type;
    using value_type = key_value_pair_view;

    view() = default;
    view(view && nt) = default;

    native_handle_type  native_handle() { return handle_.get(); }

    struct iterator
    {
        using value_type = key_value_pair_view;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(const iterator & ) = default;
        iterator(const native_iterator &native_handle) : iterator_(native_handle) {}

        iterator & operator++()
        {
            iterator_ = detail::next(iterator_);
            return *this;
        }

        iterator operator++(int)
        {
            auto last = *this;
            iterator_ = detail::next(iterator_);
            return last;
        }


        const key_value_pair_view operator*() const
        {
            return key_value_pair_view(detail::dereference(iterator_));
        }

        std::optional<key_value_pair_view> operator->() const
        {
            return key_value_pair_view(detail::dereference(iterator_));
        }

      friend bool operator==(const iterator & l, const iterator & r) {return l.iterator_ == r.iterator_;}
      friend bool operator!=(const iterator & l, const iterator & r) {return l.iterator_ != r.iterator_;}

      private:
        environment::native_iterator iterator_;
    };

    iterator begin() const {return iterator(handle_.get());}
    iterator   end() const {return iterator(detail::find_end(handle_.get()));}

 private:

  std::unique_ptr<typename remove_pointer<native_handle_type>::type,
                    detail::native_handle_deleter> handle_{environment::detail::load_native_handle()};
};


#if ASIO_HAS_FILESYSTEM

template<typename Environment = view>
inline asio::filesystem::path home(Environment && env = view())
{
  auto find_key = [&](key_view ky) -> value
  {
    const auto itr = std::find_if(std::begin(env), std::end(env),
                                  [&](key_value_pair vp)
                                  {
                                    auto tmp =  vp.key_view() == ky;
                                    if (tmp)
                                      return true;
                                    else
                                      return false;
                                  });
    if (itr != nullptr)
      return itr->value_view();
    else
      return value_view();
  };
#if defined(ASIO_WINDOWS)
  return find_key(L"HOMEDRIVE") + find_key(L"HOMEPATH");
#else
  return find_key(L"HOME");
#endif

}

template<typename Environment = view>
inline asio::filesystem::path find_executable(asio::filesystem::path name,
                                             Environment && env = view())
{
    auto find_key = [&](key_view ky) -> value_view
                    {
                        const auto itr = std::find_if(std::begin(env), std::end(env),
                                            [&](key_value_pair vp)
                                            {
                                                auto tmp =  vp.key_view() == ky;
                                                if (tmp)
                                                    return true;
                                                else
                                                    return false;
                                            });
                        if (itr != nullptr)
                          return itr->value_view();
                        else
                          return value_view();
                    };

#if defined(ASIO_WINDOWS)
    auto path = find_key(L"PATH");
    auto pathext = find_key(L"PATHEXT");
    for (auto pp_view : path)
        for (auto ext : pathext)
        {
            asio::filesystem::path nm(name);
            nm += ext;

            auto p = asio::filesystem::path(pp_view) / nm;

            error_code ec;
            bool file = asio::filesystem::is_regular_file(p, ec);
            if (!ec && file && SHGetFileInfoW(p.native().c_str(), 0,0,0, SHGFI_EXETYPE))
                return p;


        }
#else
    auto path = find_key("PATH");
    for (auto pp_view : path)
    {
        auto p = asio::filesystem::path(pp_view) / name;
        error_code ec;
        bool file = asio::filesystem::is_regular_file(p, ec);
        if (!ec && file && ::access(p.c_str(), X_OK) == 0)
            return p;
    }
#endif
    return {};
}

#endif

inline value get(const key & k, error_code & ec) { return detail::get(k.c_str(), ec);}
inline value get(const key & k)
{
  error_code ec;
  auto tmp = detail::get(k.c_str(), ec);
  asio::detail::throw_error(ec, "environment::get");
  return tmp;
}

inline value get(basic_cstring_view<char_type, key_char_traits<char_type>> k, error_code & ec)
{
  return detail::get(k, ec);
}
inline value get(basic_cstring_view<char_type, key_char_traits<char_type>> k)
{
  error_code ec;
  auto tmp = detail::get(k, ec);
  asio::detail::throw_error(ec, "environment::get");
  return tmp;
}


inline value get(const char_type * c, error_code & ec) { return detail::get(c, ec);}
inline value get(const char_type * c)
{
  error_code ec;
  auto tmp = detail::get(c, ec);
  asio::detail::throw_error(ec, "environment::get");
  return tmp;
}

inline void set(const key & k, value_view vw, error_code & ec) { detail::set(k, vw, ec);}
inline void set(const key & k, value_view vw)
{
  error_code ec;
  detail::set(k, vw, ec);
  asio::detail::throw_error(ec, "environment::set");
}

inline void set(basic_cstring_view<char_type, key_char_traits<char_type>> k, value_view vw, error_code & ec) { detail::set(k, vw, ec);}
inline void set(basic_cstring_view<char_type, key_char_traits<char_type>> k, value_view vw)
{
  error_code ec;
  detail::set(k, vw, ec);
  asio::detail::throw_error(ec, "environment::set");
}


inline void set(const char * k, ASIO_BASIC_CSTRING_VIEW_PARAM(char, value_char_traits<char>) vw, error_code & ec) { detail::set(k, vw, ec);}
inline void set(const char * k, ASIO_BASIC_CSTRING_VIEW_PARAM(char, value_char_traits<char>) vw)
{
  error_code ec;
  detail::set(k, vw, ec);
  asio::detail::throw_error(ec, "environment::set");
}


inline void unset(const key & k, error_code & ec) { detail::unset(k, ec);}
inline void unset(const key & k)
{
  error_code ec;
  detail::unset(k, ec);
  asio::detail::throw_error(ec, "environment::unset");
}

inline void unset(basic_cstring_view<char_type, key_char_traits<char_type>> k, error_code & ec)
{
  detail::unset(k, ec);
}
inline void unset(basic_cstring_view<char_type, key_char_traits<char_type>> k)
{
  error_code ec;
  detail::unset(k, ec);
  asio::detail::throw_error(ec, "environment::unset");
}


inline void unset(const char_type * c, error_code & ec) { detail::unset(c, ec);}
inline void unset(const char_type * c)
{
  error_code ec;
  detail::unset(c, ec);
  asio::detail::throw_error(ec, "environment::unset");
}
}
}

namespace std
{


template<>
struct tuple_size<asio::environment::key_value_pair> : integral_constant<std::size_t, 2u> {};

template<>
struct tuple_element<0u, asio::environment::key_value_pair> {using type = asio::environment::key_view;};

template<>
struct tuple_element<1u, asio::environment::key_value_pair> {using type = asio::environment::value_view;};

template<>
struct tuple_size<asio::environment::key_value_pair_view> : integral_constant<std::size_t, 2u> {};

template<>
struct tuple_element<0u, asio::environment::key_value_pair_view> {using type = asio::environment::key_view;};

template<>
struct tuple_element<1u, asio::environment::key_value_pair_view> {using type = asio::environment::value_view;};


}


#include "asio/detail/pop_options.hpp"

#endif //ASIO_THIS_PROCESS_HPP
