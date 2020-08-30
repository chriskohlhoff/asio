//
// detail/winrt_utils.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2020 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WINRT_UTILS_HPP
#define ASIO_DETAIL_WINRT_UTILS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include <codecvt>
#include <cstdlib>
#include <future>
#include <locale>
#include <robuffer.h>
#include <winrt/Windows.Storage.Streams.h>
#include <winrt/Windows.Networking.h>
#include <wrl/implements.h>
#include "asio/buffer.hpp"
#include "asio/error_code.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/socket_ops.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {
namespace winrt_utils {

inline winrt::hstring string(const char* from)
{
  std::wstring tmp(from, from + std::strlen(from));
  return winrt::hstring(tmp);
}

inline winrt::hstring string(const std::string& from)
{
  std::wstring tmp(from.begin(), from.end());
  return winrt::hstring(tmp);
}

inline std::string string(winrt::hstring from)
{
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(from.data());
}

inline winrt::hstring string(unsigned short from)
{
  return string(std::to_string(from));
}

template <typename T>
inline winrt::hstring string(const T& from)
{
  return string(from.to_string());
}

inline int integer(winrt::hstring from)
{
  return _wtoi(from.data());
}

template <typename T>
inline winrt::Windows::Networking::HostName host_name(const T& from)
{
  return winrt::Windows::Networking::HostName((string)(from));
}

template <typename ConstBufferSequence>
inline winrt::Windows::Storage::Streams::IBuffer buffer_dup(
    const ConstBufferSequence& buffers)
{
  using asio::buffer_size;
  std::size_t size = buffer_size(buffers);
  auto b = winrt::make<winrt::Windows::Storage::Streams::Buffer>(size);
  byte* bytes = nullptr;

  b.as<Windows::Storage::Streams::IBufferByteAccess>()->Buffer(&bytes);
  asio::buffer_copy(asio::buffer(bytes, size), buffers);
  b.Length(size);
  return b;
}

} // namespace winrt_utils
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_WINRT_UTILS_HPP
