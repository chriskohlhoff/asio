//
// windows/connect_pipe.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern klemens dot morgenstern at gmx dot net
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WINDOWS_CONNECT_PIPE_HPP
#define ASIO_WINDOWS_CONNECT_PIPE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_WINDOWS_RANDOM_ACCESS_HANDLE) \
  || defined(ASIO_HAS_WINDOWS_STREAM_HANDLE) \
  || defined(GENERATING_DOCUMENTATION)

#if defined(ASIO_HAS_MOVE)
# include <utility>
#endif // defined(ASIO_HAS_MOVE)

#include "asio/detail/push_options.hpp"

#if defined(__cpp_lib_filesystem)
#include <filesystem>
#endif

namespace asio {

template<typename Executor>
struct basic_pipe_read_end;

template<typename Executor>
struct basic_pipe_write_end;


namespace detail {
namespace windows {

inline std::wstring make_pipe_name()
{
    std::wstring name = L"\\\\.\\pipe\\boost_asio_auto_pipe_";

    DWORD pid = ::GetCurrentProcessId();

    static std::atomic_size_t cnt{0};
    name += std::to_wstring(pid);
    name += L"_";
    name += std::to_wstring(cnt++);

    return name;
}

template<typename Executor1, typename Executor2>
inline void connect_pipe(basic_pipe_read_end<Executor1> &read,
                         basic_pipe_write_end<Executor2> &write,
                         error_code &ec) ASIO_NOEXCEPT
{

    const std::wstring name = detail::windows::make_pipe_name();
    HANDLE source = ::CreateNamedPipeW(name.c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED, 0, 1, 8192, 8192,
                                       0, NULL);
    if (source == INVALID_HANDLE_VALUE) {
        ec.assign(GetLastError(), system_category());
        return;
    }
    read.assign(source, ec);

    HANDLE sink = CreateFileW(name.c_str(), GENERIC_WRITE, 0, 0, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, 0);
    if (sink == INVALID_HANDLE_VALUE) // GENERIC_WRITE
    {
        ec.assign(GetLastError(), system_category());
        return;
    }
    write.assign(sink, ec);
}

template<typename Executor1, typename Executor2>
inline void connect_pipe(basic_pipe_read_end<Executor1> &read,
                         basic_pipe_write_end<Executor2> &write)
{
    error_code ec;
    connect_pipe(read, write, ec);
    if (ec)
        detail::throw_error(ec, "connect_pipe failed");
}

#if defined(__cpp_lib_filesystem)

template<typename Executor1, typename Executor2>
inline void connect_pipe(const std::filesystem::path& filename,
                         basic_pipe_read_end<Executor1> & read,
                         basic_pipe_write_end<Executor2> & write,
                         error_code & ec) ASIO_NOEXCEPT
{
    const HANDLE proc = GetCurrentProcess();
    HANDLE pp = ::CreateNamedPipeW(
            filename.c_str(),
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED, 0, 1, 8192, 8192, 0, NULL);


    if (pp == INVALID_HANDLE_VALUE)
    {
        ec.assign(GetLastError(), system_category());
        return ;
    }

    HANDLE source;
    if (DuplicateHandle(proc, pp, proc, &source, 0, TRUE, PIPE_ACCESS_INBOUND)) // GENERIC_READ
    {
        CloseHandle(pp);
        ec.assign(GetLastError(), system_category());
        return ;
    }

    HANDLE sink;
    if (DuplicateHandle(proc, pp, proc, &sink, 0, TRUE, PIPE_ACCESS_OUTBOUND)) // GENERIC_WRITE
    {
        CloseHandle(pp);
        CloseHandle(source);
        ec.assign(GetLastError(), system_category());
        return ;
    }

    read.assign(source);
    write.assign(sink);
}

template<typename Executor1, typename Executor2>
inline void connect_pipe(const std::filesystem::path& filename,
                         basic_pipe_read_end<Executor1> & read,
                         basic_pipe_write_end<Executor2> & write)
{
    error_code ec;
    connect_pipe(filename, read, write, ec);
    if (ec)
        detail::throw_error(ec, "create_pipe(name) failed");
}

#endif

} // namespace windows
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR)
//   || defined(GENERATING_DOCUMENTATION)

#endif //ASIO_POSIX_CONNECT_PIPE_HPP
