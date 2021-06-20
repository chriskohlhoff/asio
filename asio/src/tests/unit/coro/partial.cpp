
#include <asio/experimental/coro/partial.hpp>
#include <asio/io_context.hpp>
#include "../unit_test.hpp"

using namespace asio::experimental;

namespace coro
{

void partial()
{
    asio::io_context ctx;
    bool ran = false;
    auto p = detail::post_coroutine(ctx, [&]{ran = true;});
    ASIO_CHECK(!ran);
    p.resume();
    ASIO_CHECK(!ran);
    ctx.run();
    ASIO_CHECK(ran);
}

}



ASIO_TEST_SUITE
(
    "coro",
    ASIO_TEST_CASE(coro::partial)
)