//
// channel.cpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/channel.hpp"

#include "asio/io_service.hpp"
#include "unit_test.hpp"

#if defined(ASIO_HAS_BOOST_BIND)
# include <boost/bind.hpp>
#else // defined(ASIO_HAS_BOOST_BIND)
# include <functional>
#endif // defined(ASIO_HAS_BOOST_BIND)

namespace channel_test {

#if defined(ASIO_HAS_BOOST_BIND)
namespace bindns = boost;
#else // defined(ASIO_HAS_BOOST_BIND)
namespace bindns = std;
using std::placeholders::_1;
#endif

void put_handler(asio::error_code ec,
    asio::error_code* out_ec)
{
  *out_ec = ec;
}

//------------------------------------------------------------------------------
// channel of int, 0-sized buffer

void int_buffer0_sync_sync_test()
{
  asio::io_service io_service;
  asio::channel<int> ch(io_service, 0);
  asio::error_code ec;

  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());

  ch.put(1, ec);
  ASIO_CHECK(ec == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  int value = ch.get(ec);
  ASIO_CHECK(ec == asio::error::would_block);
  ASIO_CHECK(value == 0);
  ASIO_CHECK(!ch.ready());

  ch.close();
  ASIO_CHECK(!ch.is_open());
  ASIO_CHECK(ch.ready());

  ch.put(1, ec);
  ASIO_CHECK(ec == asio::error::broken_pipe);
  ASIO_CHECK(ch.ready());

  value = ch.get(ec);
  ASIO_CHECK(ec == asio::error::broken_pipe);
  ASIO_CHECK(value == 0);
  ASIO_CHECK(ch.ready());

  ch.reset();
  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());

  ch.put(1, ec);
  ASIO_CHECK(ec == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  value = ch.get(ec);
  ASIO_CHECK(ec == asio::error::would_block);
  ASIO_CHECK(value == 0);
  ASIO_CHECK(!ch.ready());
}

void int_buffer0_async_sync_test()
{
  asio::io_service io_service;
  asio::channel<int> ch(io_service, 0);
  asio::error_code put_ec_1, put_ec_2, get_ec;

  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());

  // put 1 / get 1

  put_ec_1 = asio::error::would_block;
  ch.async_put(1, bindns::bind(&put_handler, _1, &put_ec_1));
  ASIO_CHECK(put_ec_1 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  int value = ch.get(get_ec);
  ASIO_CHECK(!get_ec);
  ASIO_CHECK(value == 1);
  ASIO_CHECK(!ch.ready());

  value = ch.get(get_ec);
  ASIO_CHECK(get_ec == asio::error::would_block);
  ASIO_CHECK(value == 0);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(!put_ec_1);

  // put 2 / get 2

  put_ec_1 = asio::error::would_block;
  ch.async_put(1, bindns::bind(&put_handler, _1, &put_ec_1));
  ASIO_CHECK(put_ec_1 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  put_ec_2 = asio::error::would_block;
  ch.async_put(2, bindns::bind(&put_handler, _1, &put_ec_2));
  ASIO_CHECK(put_ec_2 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  value = ch.get(get_ec);
  ASIO_CHECK(!get_ec);
  ASIO_CHECK(value == 1);
  ASIO_CHECK(ch.ready());

  value = ch.get(get_ec);
  ASIO_CHECK(!get_ec);
  ASIO_CHECK(value == 2);
  ASIO_CHECK(!ch.ready());

  value = ch.get(get_ec);
  ASIO_CHECK(get_ec == asio::error::would_block);
  ASIO_CHECK(value == 0);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(!put_ec_1);
  ASIO_CHECK(!put_ec_2);

  // cancel

  put_ec_1 = asio::error::would_block;
  ch.async_put(1, bindns::bind(&put_handler, _1, &put_ec_1));
  ASIO_CHECK(put_ec_1 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  ch.cancel();
  ASIO_CHECK(!ch.ready());

  value = ch.get(get_ec);
  ASIO_CHECK(get_ec == asio::error::would_block);
  ASIO_CHECK(value == 0);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(put_ec_1 == asio::error::operation_aborted);
  ASIO_CHECK(!ch.ready());

  // put none / close

  ch.close();
  ASIO_CHECK(!ch.is_open());
  ASIO_CHECK(ch.ready());

  value = ch.get(get_ec);
  ASIO_CHECK(get_ec == asio::error::broken_pipe);
  ASIO_CHECK(value == 0);
  ASIO_CHECK(ch.ready());

  ch.reset();
  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());

  // put 1 / close

  put_ec_1 = asio::error::would_block;
  ch.async_put(1, bindns::bind(&put_handler, _1, &put_ec_1));
  ASIO_CHECK(put_ec_1 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  ch.close();
  ASIO_CHECK(!ch.is_open());
  ASIO_CHECK(ch.ready());

  value = ch.get(get_ec);
  ASIO_CHECK(!get_ec);
  ASIO_CHECK(value == 1);
  ASIO_CHECK(ch.ready());

  value = ch.get(get_ec);
  ASIO_CHECK(get_ec == asio::error::broken_pipe);
  ASIO_CHECK(value == 0);
  ASIO_CHECK(ch.ready());

  ch.reset();
  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());
}

//------------------------------------------------------------------------------
// channel of void, 0-sized buffer

void void_buffer0_sync_sync_test()
{
  asio::io_service io_service;
  asio::channel<void> ch(io_service, 0);
  asio::error_code ec;

  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());

  ch.put(ec);
  ASIO_CHECK(ec == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  ch.get(ec);
  ASIO_CHECK(ec == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  ch.close();
  ASIO_CHECK(!ch.is_open());
  ASIO_CHECK(ch.ready());

  ch.put(ec);
  ASIO_CHECK(ec == asio::error::broken_pipe);
  ASIO_CHECK(ch.ready());

  ch.get(ec);
  ASIO_CHECK(ec == asio::error::broken_pipe);
  ASIO_CHECK(ch.ready());

  ch.reset();
  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());

  ch.put(ec);
  ASIO_CHECK(ec == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  ch.get(ec);
  ASIO_CHECK(ec == asio::error::would_block);
  ASIO_CHECK(!ch.ready());
}

} // namespace channel_test

ASIO_TEST_SUITE
(
  "channel",
  ASIO_TEST_CASE(channel_test::int_buffer0_sync_sync_test)
  ASIO_TEST_CASE(channel_test::int_buffer0_async_sync_test)
  ASIO_TEST_CASE(channel_test::void_buffer0_sync_sync_test)
)
