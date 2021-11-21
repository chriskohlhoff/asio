//
// process/environment.hpp
// ~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//


#ifndef ASIO_PROCESS_ENVIRONMENT_HPP
#define ASIO_PROCESS_ENVIRONMENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/process/default_launcher.hpp"
#include "asio/process/environment.hpp"
#include "asio/detail/codecvt.hpp"
#include "asio/detail/string_view.hpp"
#include "asio/detail/push_options.hpp"

namespace asio
{

struct process_environment
{

#if defined(ASIO_WINDOWS)


  template<typename Args>
  void build_env(Args && args, ASIO_BASIC_STRING_VIEW_PARAM(char) rs)
  {
    std::vector<decltype(rs)> vec;
  //  vec.reserve(std::end(args) - std::begin(args));
    std::size_t length = 0u;
    for (decltype(rs) v : std::forward<Args>(args))
    {
      vec.push_back(v);
      length += v.size() + 1u;
    }
    length ++ ;

    ascii_env.resize(length);

    auto itr = ascii_env.begin();
    for (const auto & v : vec )
    {
      itr = std::copy(v.begin(), v.end(), itr);
      *(itr++) = '\0';
    }
    ascii_env.back() = '\0';
  }
  template<typename Args>
  void build_env(Args && args, ASIO_BASIC_STRING_VIEW_PARAM(wchar_t) rs)
  {
    std::vector<decltype(rs)> vec;
//    vec.reserve(std::end(args) - std::begin(args));
    std::size_t length = 0u;
    for (decltype(rs) v : std::forward<Args>(args))
    {
      vec.push_back(v);
      length += v.size() + 1u;
    }
    length ++ ;

    unicode_env.resize(length);

    auto itr = unicode_env.begin();
    for (const auto & v : vec )
    {
      itr = std::copy(v.begin(), v.end(), itr);
      *(itr++) = L'\0';
    }
    unicode_env.back() = L'\0';
  }

  template<typename Args>
  void build_env(Args && args, ASIO_BASIC_STRING_VIEW_PARAM(char16_t) rs)
  {
    std::vector<std::wstring> env;
    env.reserve(std::end(args) - std::begin(args));
    for (decltype(rs) v : std::forward<Args>(args))
      env.push_back(detail::convert_chars(v.data(), v.data() + v.size(), L' '));
    build_env(env, L"");

  }
  template<typename Args>
  void build_env(Args && args, ASIO_BASIC_STRING_VIEW_PARAM(char32_t) rs)
  {
    std::vector<std::wstring> env;
    env.reserve(std::end(args) - std::begin(args));
    for (decltype(rs) v : std::forward<Args>(args))
      env.push_back(detail::convert_chars(v.data(), v.data() + v.size(), L' '));
    build_env(env, L"");

  }

#if ASIO_HAS_CHAR8_T
  template<typename Args>
  void build_env(Args && args, ASIO_BASIC_STRING_VIEW_PARAM(char8_t) rs)
  {
    std::vector<std::wstring> env;
    env.reserve(std::end(args) - std::begin(args));
    for (decltype(rs) v : std::forward<Args>(args))
      env.push_back(detail::convert_chars(v.data(), v.data() + v.size(), L' '));
    build_env(env, L"");
  }
#endif

  process_environment(std::initializer_list<cstring_view> sv)  { build_env(sv,  ""); }
  process_environment(std::initializer_list<wcstring_view> sv) { build_env(sv, L""); }

  template<typename Args>
  process_environment(Args && args)
  {
    if (std::begin(args) != std::end(args))
      build_env(std::forward<Args>(args), *std::begin(args));
  }


  std::vector<char> ascii_env;
  std::vector<wchar_t> unicode_env;


  error_code on_setup(windows::default_launcher & launcher, const filesystem::path &, const std::wstring &)
  {
    if (!unicode_env.empty())
    {
      launcher.creation_flags |= CREATE_UNICODE_ENVIRONMENT ;
      launcher.environment = unicode_env.data();
    }
    else if (!ascii_env.empty())
      launcher.environment = ascii_env.data();

    return error_code {};
  };

#else

  template<typename Args>
  static
  std::vector<const char *> build_env(Args && args,
                                      typename enable_if<
                                              std::is_convertible<
                                                      decltype(*std::begin(std::declval<Args>())),
                                                      ASIO_CSTRING_VIEW>::value>::type * = nullptr)
  {
    std::vector<const char *> env;
    for (auto && e : args)
      env.push_back(e.c_str());

    env.push_back(nullptr);
    return env;
  }

  template<typename Args>
  std::vector<const char *> build_env(Args && args,
                                      typename enable_if<
                                              !std::is_convertible<
                                                      decltype(*std::begin(std::declval<Args>())),
                                                      ASIO_CSTRING_VIEW>::value>::type * = nullptr)
  {
    std::vector<const char *> env;

    using char_type = typename decay<decltype((*std::begin(std::declval<Args>()))[0])>::type;
    for (ASIO_BASIC_STRING_VIEW_PARAM(char_type)  arg : args)
      env_buffer.push_back(detail::convert_chars(arg.data(), arg.data() + arg.size(), ' '));

    for (auto && e : env_buffer)
      env.push_back(e.c_str());
    env.push_back(nullptr);
    return env;
  }


  process_environment(std::initializer_list<cstring_view> sv) : env{build_env(sv)}  {  }

  template<typename Args>
  process_environment(Args && args) : env(build_env(std::forward<Args>(args)))
  {
  }


  error_code on_setup(posix::default_launcher & launcher, const filesystem::path &, const char * const *)
  {
    launcher.env = env.data();
    return error_code{};
  };

  std::vector<const char *> env;
  std::vector<std::string> env_buffer;

#endif

};

}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_PROCESS_ENVIRONMENT_HPP
