//
// posix/connect_pipe.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern klemens dot morgenstern at gmx dot net
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_POSIX_CONNECT_PIPE_HPP
#define ASIO_POSIX_CONNECT_PIPE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
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

namespace detail
{
namespace posix
{

template<typename Executor1, typename Executor2>
inline void connect_pipe(basic_pipe_read_end <Executor1> &read,
                         basic_pipe_write_end <Executor2> &write,
                         error_code &ec) ASIO_NOEXCEPT
{
    int p[2];
    if (::pipe(p) == -1)
        ec.assign(errno, system_category());
    else
    {
        read.assign(p[0], ec);
        write.assign(p[1], ec);
    }
}

template<typename Executor1, typename Executor2>
inline void connect_pipe(basic_pipe_read_end <Executor1> &read,
                         basic_pipe_write_end <Executor2> &write)
{
    error_code ec;
    connect_pipe(read, write, ec);
    if (ec)
        detail::throw_error(ec, "create_pipe failed");
}

#if defined(__cpp_lib_filesystem)

template<typename Executor1, typename Executor2>
inline void connect_pipe(const std::filesystem::path &filename,
                         basic_pipe_read_end <Executor1> &read,
                         basic_pipe_write_end <Executor2> &write,
                         error_code &ec) ASIO_NOEXCEPT
{
    int p = ::mkfifo(filename.c_str(), O_RDWR);
    if (p == -1)
    {
        ec.assign(errno, system_category());
        return;
    }

    int read_fd = open(filename.c_str(), O_RDONLY);

    if (read_fd == -1)
    {
        ec.assign(errno, system_category());
        return;
    }

    int write_fd = open(filename.c_str(), O_WRONLY);

    if (write_fd == -1)
    {
        ec.assign(errno, system_category());
        ::close(read_fd);
        return;
    }

    read.assign(read_fd, ec);
    write.assign(write_fd, ec);
}

template<typename Executor1, typename Executor2>
inline void connect_pipe(const std::filesystem::path &filename,
                         basic_pipe_read_end <Executor1> &read,
                         basic_pipe_write_end <Executor2> &write)
{
    error_code ec;
    connect_pipe(filename, read, write, ec);
    if (ec)
        detail::throw_error(ec, "create_pipe(name) failed");
}

#endif


} // namespace posix
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR)
//   || defined(GENERATING_DOCUMENTATION)

#endif //ASIO_POSIX_CONNECT_PIPE_HPP
