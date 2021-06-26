//
// posix/devices.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern klemens dot morgenstern at gmx dot net
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_POSIX_DEVICES_HPP
#define ASIO_POSIX_DEVICES_HPP


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

namespace asio {

template<typename Executor>
struct basic_pipe_read_end;

template<typename Executor>
struct basic_pipe_write_end;

namespace detail {
namespace posix {

template<typename Executor>
inline void open_null_reader(basic_pipe_read_end <Executor> &pipe_, error_code &ec) ASIO_NOEXCEPT
{
    int fd = ::open("/dev/null", O_RDONLY, 0660);
    if (fd == -1)
        ec.assign(errno, system_category());
    else
        pipe_.assign(fd, ec);
}

template<typename Executor>
inline void open_null_reader(basic_pipe_read_end <Executor> &pipe_)
{
    error_code ec;
    open_null_reader(pipe_, ec);
    if (ec)
        detail::throw_error(ec, "open(/dev/null)");

}

template<typename Executor>
inline void open_null_writer(basic_pipe_write_end <Executor> &pipe_, error_code &ec) ASIO_NOEXCEPT
{
    int fd = ::open("/dev/null", O_WRONLY, 0660);
    if (fd == -1)
        ec.assign(errno, system_category());
    else
        pipe_.assign(fd, ec);
}

template<typename Executor>
inline void open_null_writer(basic_pipe_write_end <Executor> &pipe_)
{
    error_code ec;
    open_null_writer(pipe_, ec);
    if (ec)
        detail::throw_error(ec, "open(/dev/null)");

}

template<typename Executor>
inline void open_stdin(basic_pipe_read_end <Executor> &pipe_, error_code &ec) ASIO_NOEXCEPT
{
    int fd = ::dup(STDIN_FILENO);
    if (fd == -1)
        ec.assign(errno, system_category());
    else
        pipe_.assign(fd, ec);
}

template<typename Executor>
inline void open_stdin(basic_pipe_read_end <Executor> &pipe_)
{
    error_code ec;
    open_stdin(pipe_, ec);
    if (ec)
        detail::throw_error(ec, "open(/dev/null)");

}

template<typename Executor>
inline void open_stdout(basic_pipe_write_end <Executor> &pipe_, error_code &ec) ASIO_NOEXCEPT
{
    int fd = ::dup(STDOUT_FILENO);
    if (fd == -1)
        ec.assign(errno, system_category());
    else
        pipe_.assign(fd, ec);
}

template<typename Executor>
inline void open_stdout(basic_pipe_write_end <Executor> &pipe_)
{
    error_code ec;
    open_stdout(pipe_, ec);
    if (ec)
        detail::throw_error(ec, "open(/dev/null)");
}

template<typename Executor>
inline void open_stderr(basic_pipe_write_end <Executor> &pipe_, error_code &ec) ASIO_NOEXCEPT
{
    int fd = ::dup(STDERR_FILENO);
    if (fd == -1)
        ec.assign(errno, system_category());
    else
        pipe_.assign(fd, ec);
}

template<typename Executor>
inline void open_stderr(basic_pipe_write_end <Executor> &pipe_)
{
    error_code ec;
    open_stderr(pipe_, ec);
    if (ec)
        detail::throw_error(ec, "open(/dev/null)");
}

#if defined (ASIO_HAS_MOVE)


template<typename Executor>
inline basic_pipe_read_end <Executor> open_null_reader(Executor executor, error_code &ec) ASIO_NOEXCEPT
{
    basic_pipe_read_end<Executor> res{std::move(executor)};
    open_null_reader(res, ec);
    return res;
}

template<typename Executor>
inline basic_pipe_read_end <Executor> open_null_reader(Executor executor)
{
    basic_pipe_read_end<Executor> res{std::move(executor)};
    open_null_reader(res);
    return res;
}

template<typename Executor>
inline basic_pipe_write_end <Executor> open_null_writer(Executor executor, error_code &ec) ASIO_NOEXCEPT
{
    basic_pipe_read_end<Executor> res{std::move(executor)};
    open_null_writer(res, ec);
    return res;
}

template<typename Executor>
inline basic_pipe_write_end <Executor> open_null_writer(Executor executor)
{
    basic_pipe_read_end<Executor> res{std::move(executor)};
    open_null_writer(res);
    return res;
}

template<typename Executor>
inline basic_pipe_read_end <Executor> open_stdin(Executor executor, error_code &ec) ASIO_NOEXCEPT
{
    basic_pipe_read_end<Executor> res{std::move(executor)};
    open_stdin(res, ec);
    return res;

}

template<typename Executor>
inline basic_pipe_read_end <Executor> open_stdin(Executor executor)
{
    basic_pipe_read_end<Executor> res{std::move(executor)};
    open_stdin(res);
    return res;
}

template<typename Executor>
inline basic_pipe_write_end <Executor> open_stdout(Executor executor, error_code &ec) ASIO_NOEXCEPT
{
    basic_pipe_write_end<Executor> res{std::move(executor)};
    open_stdout(res, ec);
    return res;
}

template<typename Executor>
inline basic_pipe_write_end <Executor> open_stdout(Executor executor)
{
    basic_pipe_write_end<Executor> res{std::move(executor)};
    open_stdout(res);
    return res;

}

template<typename Executor>
inline basic_pipe_write_end <Executor> open_stderr(Executor executor, error_code &ec) ASIO_NOEXCEPT
{
    basic_pipe_write_end<Executor> res{std::move(executor)};
    open_stderr(res, ec);
    return res;

}

template<typename Executor>
inline basic_pipe_write_end <Executor> open_stderr(Executor executor)
{
    basic_pipe_write_end<Executor> res{std::move(executor)};
    open_stderr(res);
    return res;

}


#endif

} // namespace posix
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR)
//   || defined(GENERATING_DOCUMENTATION)

#endif //ASIO_DEVICES_HPP
