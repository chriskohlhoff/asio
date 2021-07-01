//
// coro/cancel.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//


#include <iostream>
#include <asio/experimental/coro.hpp>
#include <asio/io_context.hpp>
#include <boost/scope_exit.hpp>
#include <asio/steady_timer.hpp>
#include "../unit_test.hpp"

using namespace asio::experimental;

namespace coro
{


void coro_simple_cancel()
{
    asio::io_context ctx;

    auto k = [](asio::io_context& ) noexcept -> asio::experimental::coro<void() noexcept, std::error_code>
        {
            asio::steady_timer timer{co_await asio::this_coro::executor, std::chrono::seconds(1)};

            ASIO_CHECK(!(co_await asio::this_coro::cancellation_state).cancelled());
            auto ec = co_await timer;
            ASIO_CHECK((co_await asio::this_coro::cancellation_state).cancelled());

            co_return ec;
        }(ctx);

    std::error_code res_ec;
    k.async_resume([&](std::error_code ec) {res_ec = ec;});
    asio::post(ctx, [&]{k.cancel();});

    ASIO_CHECK(!res_ec);

    ctx.run();

    ASIO_CHECK(res_ec == asio::error::operation_aborted);
}

void coro_throw_cancel()
{
    asio::io_context ctx;

    auto k = [](asio::io_context& ) -> asio::experimental::coro<void() , void>
    {
        asio::steady_timer timer{co_await asio::this_coro::executor, std::chrono::seconds(1)};
        co_await timer;
    }(ctx);

    std::exception_ptr res_ex;
    k.async_resume([&](std::exception_ptr ex) {res_ex = ex;});
    asio::post(ctx, [&]{k.cancel();});

    ASIO_CHECK(!res_ex);

    ctx.run();

    ASIO_CHECK(res_ex);
    try {
        std::rethrow_exception(res_ex);
    }
    catch (std::system_error & se)
    {
        ASIO_CHECK(se.code() == asio::error::operation_aborted);
    }
}

void coro_simple_cancel_nested()
{
    asio::io_context ctx;

    auto k = [](asio::io_context&, int & cnt ) noexcept -> asio::experimental::coro<void() noexcept, std::error_code>
    {
        asio::steady_timer timer{co_await asio::this_coro::executor, std::chrono::milliseconds(100)};

        ASIO_CHECK(!(co_await asio::this_coro::cancellation_state).cancelled());
        auto ec = co_await timer;
        cnt++;
        ASIO_CHECK((co_await asio::this_coro::cancellation_state).cancelled());

        co_return ec;
    };

    int cnt = 0;
    auto kouter = [&](asio::io_context& ctx, int & cnt ) noexcept -> asio::experimental::coro<std::error_code() noexcept, std::error_code>
    {
        ASIO_CHECK(cnt == 0);
        co_yield co_await k(ctx, cnt);
        ASIO_CHECK(cnt == 1);
        auto ec = co_await k(ctx, cnt);
        ASIO_CHECK(cnt == 2);
        co_return ec;
    }(ctx, cnt);

    std::error_code res_ec;
    kouter.async_resume([&](std::error_code ec) {res_ec = ec;});
    asio::post(ctx, [&]{kouter.cancel();});
    ASIO_CHECK(!res_ec);
    ctx.run();
    ASIO_CHECK(res_ec == asio::error::operation_aborted);

    ctx.restart();
    res_ec = {};
    kouter.async_resume([&](std::error_code ec) {res_ec = ec;});
    asio::post(ctx, [&]{kouter.cancel();});
    ASIO_CHECK(!res_ec);
    ctx.run();
    ASIO_CHECK(res_ec == asio::error::operation_aborted);
    ASIO_CHECK(cnt == 2);
}

}

ASIO_TEST_SUITE
(
    "coro",
    ASIO_TEST_CASE(::coro::coro_simple_cancel)
    ASIO_TEST_CASE(::coro::coro_throw_cancel)
    ASIO_TEST_CASE(::coro::coro_simple_cancel_nested)
)