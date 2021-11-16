//
// process/stdio.hpp
// ~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//


#ifndef ASIO_PROCESS_STDIO_HPP
#define ASIO_PROCESS_STDIO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/process/default_launcher.hpp"
#include "asio/detail/push_options.hpp"

namespace asio
{

namespace detail
{
#if defined(ASIO_WINDOWS)

extern "C" intptr_t _get_osfhandle(int fd);

struct handle_closer
{
  handle_closer() = default;
  handle_closer(bool close) : close(close) {}
  handle_closer(DWORD flags) : close(false), flags{flags} {}


  void operator()(HANDLE h) const
  {
    if (close)
      ::CloseHandle(h);
    else if (flags != 0xFFFFFFFF)
      ::SetHandleInformation(h, flags, flags);

  }

  bool close{false};
  DWORD flags{0xFFFFFFFF};
};

template<DWORD Io>
struct process_io_binding
{
  HANDLE prepare()
  {
    auto hh =  h.get();
    ::SetHandleInformation(hh, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
    return hh;
  }

  std::unique_ptr<void, handle_closer> h{::GetStdHandle(Io), false};

  static DWORD get_flags(HANDLE h)
  {
    DWORD res;
    if (!::GetHandleInformation(h, &res))
      detail::throw_error(error_code(::GetLastError(), system_category()));
    return res;
  }

  process_io_binding() = default;

  template<typename Stream>
  process_io_binding(Stream && str, decltype(std::declval<Stream>().native_handle()) = nullptr)
      : process_io_binding(str.native_handle())
  {}

  process_io_binding(FILE * f) : process_io_binding(_get_osfhandle(_fileno(f))) {}
  process_io_binding(HANDLE h) : h{h, get_flags(h)} {}
  process_io_binding(std::nullptr_t) : process_io_binding(filesystem::path("NUL")) {}
  process_io_binding(const filesystem::path & pth)
    : h(::CreateFileW(
        pth.c_str(),
        Io == STD_INPUT_HANDLE ? GENERIC_READ : GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
        ), true)
  {
  }


};

typedef process_io_binding<STD_INPUT_HANDLE>  process_input_binding;
typedef process_io_binding<STD_OUTPUT_HANDLE> process_output_binding;
typedef process_io_binding<STD_ERROR_HANDLE>  process_error_binding;

#else

#endif

}

struct process_stdio
{
  detail::process_input_binding in;
  detail::process_output_binding out;
  detail::process_error_binding err;

#if defined(ASIO_WINDOWS)
  error_code on_setup(windows::default_launcher & launcher, const filesystem::path &, const std::wstring &)
  {

    launcher.startup_info.StartupInfo.dwFlags |= STARTF_USESTDHANDLES;
    launcher.startup_info.StartupInfo.hStdInput  = in.prepare();
    launcher.startup_info.StartupInfo.hStdOutput = out.prepare();
    launcher.startup_info.StartupInfo.hStdError  = err.prepare();
    launcher.inherit_handles = true;
    return error_code {};
  };

#endif

};

}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_PROCESS_STDIO_HPP
