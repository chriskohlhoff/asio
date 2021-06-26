//
// pipe.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/basic_pipe.hpp"
#include "asio/read.hpp"
#include "asio/write.hpp"
#include "asio/detached.hpp"

#include "unit_test.hpp"

void basic_pipe_test()
{
    asio::io_context ctx;

    asio::pipe p{ctx.get_executor()};

    std::string src = "foobar";
    std::string res;
    res.resize(src.size());
    asio::async_write(p, asio::buffer(src), asio::detached);
    asio::async_read(p, asio::buffer(res), asio::detached);

    ctx.run();

    ASIO_CHECK(res == src);

    src = "barfoo";
}


void basic_stdio_test()
{
    asio::io_context ctx;

    asio::pipe_read_end null_r{ctx.get_executor()};
    asio::open_null_reader(null_r);
    ASIO_CHECK(null_r.is_open());

    asio::pipe_write_end null_w{ctx.get_executor()};
    asio::open_null_writer(null_w);
    ASIO_CHECK(null_w.is_open());
}

namespace asio
{
template struct basic_pipe<any_io_executor>;
template struct basic_pipe_read_end<any_io_executor>;
template struct basic_pipe_write_end<any_io_executor>;
}

ASIO_TEST_SUITE
(
"basic_pipe",
ASIO_TEST_CASE(basic_pipe_test)
    ASIO_TEST_CASE(basic_stdio_test)
)
