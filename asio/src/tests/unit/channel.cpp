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
#include <list>

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
using std::placeholders::_2;
#endif

void put_handler(asio::error_code ec,
    asio::error_code* out_ec)
{
  *out_ec = ec;
}

void get_handler(asio::error_code ec, int value,
    asio::error_code* out_ec, int* out_value)
{
  *out_ec = ec;
  *out_value = value;
}

void sync_put_sync_get_test(const std::size_t capacity)
{
  asio::io_service io_service;
  asio::channel<int> ch(io_service, capacity);
  asio::error_code ec;

  ASIO_CHECK(ch.is_open());

  // put / get

  for (std::size_t max_puts = 0; max_puts < capacity * 2 + 1; ++max_puts)
  {
    const std::size_t buffered_puts = (std::min)(max_puts, capacity);

    for (std::size_t partial_puts = 0;
        partial_puts < buffered_puts; ++partial_puts)
    {
      for (std::size_t partial_gets = 0;
          partial_gets + 1 < partial_puts; ++partial_gets)
      {
        std::size_t puts = 0;
        std::size_t gets = 0;

        ASIO_CHECK(!ch.ready());

        for (; puts < partial_puts; ++puts)
        {
          ch.put(static_cast<int>(puts), ec);
          ASIO_CHECK(!ec);
          ASIO_CHECK(ch.ready());
        }

        for (; gets < partial_gets; ++gets)
        {
          ASIO_CHECK(ch.ready());
          int value = ch.get(ec);
          ASIO_CHECK(!ec);
          ASIO_CHECK(value == static_cast<int>(gets));
        }

        for (; puts < buffered_puts + partial_gets; ++puts)
        {
          ch.put(static_cast<int>(puts), ec);
          ASIO_CHECK(!ec);
          ASIO_CHECK(ch.ready());
        }

        for (; puts < max_puts + partial_gets; ++puts)
        {
          ch.put(static_cast<int>(puts), ec);
          ASIO_CHECK(ec == asio::error::would_block);
          if (capacity > 0)
            ASIO_CHECK(ch.ready());
          else
            ASIO_CHECK(!ch.ready());
        }

        for (; gets < buffered_puts + partial_gets; ++gets)
        {
          ASIO_CHECK(ch.ready());
          int value = ch.get(ec);
          ASIO_CHECK(!ec);
          ASIO_CHECK(value == static_cast<int>(gets));
        }

        for (; gets < max_puts + partial_gets + 1; ++gets)
        {
          ASIO_CHECK(!ch.ready());
          int value = ch.get(ec);
          ASIO_CHECK(ec == asio::error::would_block);
          ASIO_CHECK(value == 0);
        }
      }
    }
  }

  // put / close / put / get

  for (std::size_t max_puts = 0;
      max_puts < capacity * 2 + 1; ++max_puts)
  {
    const std::size_t buffered_puts = (std::min)(max_puts, capacity);
    std::size_t puts = 0;
    std::size_t gets = 0;

    ASIO_CHECK(!ch.ready());

    for (; puts < buffered_puts; ++puts)
    {
      ch.put(static_cast<int>(puts), ec);
      ASIO_CHECK(!ec);
      ASIO_CHECK(ch.ready());
    }

    for (; puts < max_puts; ++puts)
    {
      ch.put(static_cast<int>(puts), ec);
      ASIO_CHECK(ec == asio::error::would_block);
      if (capacity > 0)
        ASIO_CHECK(ch.ready());
      else
        ASIO_CHECK(!ch.ready());
    }

    ch.close();
    ASIO_CHECK(!ch.is_open());
    ASIO_CHECK(ch.ready());

    ch.put(1, ec);
    ASIO_CHECK(ec == asio::error::broken_pipe);
    ASIO_CHECK(ch.ready());

    for (; gets < buffered_puts; ++gets)
    {
      ASIO_CHECK(ch.ready());
      int value = ch.get(ec);
      ASIO_CHECK(!ec);
      ASIO_CHECK(value == static_cast<int>(gets));
    }

    for (; gets < max_puts + 1; ++gets)
    {
      ASIO_CHECK(ch.ready());
      int value = ch.get(ec);
      ASIO_CHECK(ec == asio::error::broken_pipe);
      if (ec != asio::error::broken_pipe)
      ASIO_CHECK(value == 0);
    }

    ch.reset();
    ASIO_CHECK(ch.is_open());
    ASIO_CHECK(!ch.ready());
  }
}

void async_put_sync_get_test(const std::size_t capacity)
{
  asio::io_service io_service;
  asio::channel<int> ch(io_service, capacity);
  asio::error_code ec;

  ASIO_CHECK(ch.is_open());

  // put / get

  for (std::size_t max_puts = 0; max_puts < capacity * 2 + 1; ++max_puts)
  {
    const std::size_t buffered_puts = (std::min)(max_puts, capacity);

    for (std::size_t partial_puts = 0;
        partial_puts < buffered_puts; ++partial_puts)
    {
      for (std::size_t partial_gets = 0;
          partial_gets + 1 < partial_puts; ++partial_gets)
      {
        std::size_t puts = 0;
        std::size_t gets = 0;
        std::list<asio::error_code> put_ecs;

        ASIO_CHECK(!ch.ready());

        for (; puts < partial_puts; ++puts)
        {
          put_ecs.push_back(asio::error::would_block);
          ch.async_put(static_cast<int>(puts),
              bindns::bind(&put_handler, _1, &put_ecs.back()));
          ASIO_CHECK(put_ecs.back() == asio::error::would_block);
          ASIO_CHECK(ch.ready());
        }

        io_service.reset();
        io_service.run();
        while (!put_ecs.empty())
        {
          ASIO_CHECK(!put_ecs.front());
          put_ecs.pop_front();
        }

        for (; gets < partial_gets; ++gets)
        {
          ASIO_CHECK(ch.ready());
          int value = ch.get(ec);
          ASIO_CHECK(!ec);
          ASIO_CHECK(value == static_cast<int>(gets));
        }

        for (; puts < buffered_puts + partial_gets; ++puts)
        {
          put_ecs.push_back(asio::error::would_block);
          ch.async_put(static_cast<int>(puts),
              bindns::bind(&put_handler, _1, &put_ecs.back()));
          ASIO_CHECK(put_ecs.back() == asio::error::would_block);
          ASIO_CHECK(ch.ready());
        }

        io_service.reset();
        io_service.run();
        while (!put_ecs.empty())
        {
          ASIO_CHECK(!put_ecs.front());
          put_ecs.pop_front();
        }

        for (; puts < max_puts + partial_gets; ++puts)
        {
          put_ecs.push_back(asio::error::would_block);
          ch.async_put(static_cast<int>(puts),
              bindns::bind(&put_handler, _1, &put_ecs.back()));
          ASIO_CHECK(put_ecs.back() == asio::error::would_block);
          ASIO_CHECK(ch.ready());
        }

        for (; gets < max_puts + partial_gets; ++gets)
        {
          ASIO_CHECK(ch.ready());
          int value = ch.get(ec);
          ASIO_CHECK(!ec);
          ASIO_CHECK(value == static_cast<int>(gets));
        }

        for (; gets < max_puts + partial_gets + 1; ++gets)
        {
          ASIO_CHECK(!ch.ready());
          int value = ch.get(ec);
          ASIO_CHECK(ec == asio::error::would_block);
          ASIO_CHECK(value == 0);
        }

        io_service.reset();
        io_service.run();
        while (!put_ecs.empty())
        {
          ASIO_CHECK(!put_ecs.front());
          put_ecs.pop_front();
        }
      }
    }
  }

  // put / close / put / get

  for (std::size_t max_puts = 0;
      max_puts < capacity * 2 + 1; ++max_puts)
  {
    const std::size_t buffered_puts = (std::min)(max_puts, capacity);
    std::size_t puts = 0;
    std::size_t gets = 0;
    std::list<asio::error_code> put_ecs;

    ASIO_CHECK(!ch.ready());

    for (; puts < buffered_puts; ++puts)
    {
      put_ecs.push_back(asio::error::would_block);
      ch.async_put(static_cast<int>(puts),
          bindns::bind(&put_handler, _1, &put_ecs.back()));
      ASIO_CHECK(put_ecs.back() == asio::error::would_block);
      ASIO_CHECK(ch.ready());
    }

    io_service.reset();
    io_service.run();
    while (!put_ecs.empty())
    {
      ASIO_CHECK(!put_ecs.front());
      put_ecs.pop_front();
    }

    for (; puts < max_puts; ++puts)
    {
      put_ecs.push_back(asio::error::would_block);
      ch.async_put(static_cast<int>(puts),
          bindns::bind(&put_handler, _1, &put_ecs.back()));
      ASIO_CHECK(put_ecs.back() == asio::error::would_block);
      ASIO_CHECK(ch.ready());
    }

    ch.close();
    ASIO_CHECK(!ch.is_open());
    ASIO_CHECK(ch.ready());

    put_ecs.push_back(asio::error::would_block);
    ch.async_put(static_cast<int>(puts),
        bindns::bind(&put_handler, _1, &put_ecs.back()));
    ASIO_CHECK(put_ecs.back() == asio::error::would_block);
    ASIO_CHECK(ch.ready());

    for (; gets < max_puts; ++gets)
    {
      ASIO_CHECK(ch.ready());
      int value = ch.get(ec);
      ASIO_CHECK(!ec);
      ASIO_CHECK(value == static_cast<int>(gets));
    }

    for (; gets < max_puts + 1; ++gets)
    {
      ASIO_CHECK(ch.ready());
      int value = ch.get(ec);
      ASIO_CHECK(ec == asio::error::broken_pipe);
      if (ec != asio::error::broken_pipe)
      ASIO_CHECK(value == 0);
    }

    io_service.reset();
    io_service.run();
    while (!put_ecs.empty())
    {
      if (put_ecs.size() > 1)
        ASIO_CHECK(!put_ecs.front());
      else
        ASIO_CHECK(put_ecs.front() == asio::error::broken_pipe);
      put_ecs.pop_front();
    }

    ch.reset();
    ASIO_CHECK(ch.is_open());
    ASIO_CHECK(!ch.ready());
  }
}

#if 0
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

void int_buffer0_sync_async_test()
{
  asio::io_service io_service;
  asio::channel<int> ch(io_service, 0);
  asio::error_code put_ec, get_ec_1, get_ec_2;
  int value_1 = 0, value_2 = 0;

  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());

  // get 1 / put 1

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  ch.put(1, put_ec);
  ASIO_CHECK(!put_ec);
  ASIO_CHECK(!ch.ready());

  ch.put(2, put_ec);
  ASIO_CHECK(put_ec == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(!get_ec_1);
  ASIO_CHECK(value_1 == 1);

  // get 2 / put 2

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  get_ec_2 = asio::error::would_block;
  value_2 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_2, &value_2));
  ASIO_CHECK(get_ec_2 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  ch.put(1, put_ec);
  ASIO_CHECK(!put_ec);
  ASIO_CHECK(!ch.ready());

  ch.put(2, put_ec);
  ASIO_CHECK(!put_ec);
  ASIO_CHECK(!ch.ready());

  ch.put(3, put_ec);
  ASIO_CHECK(put_ec == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(!get_ec_1);
  ASIO_CHECK(value_1 == 1);
  ASIO_CHECK(!get_ec_2);
  ASIO_CHECK(value_2 == 2);

  // cancel

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  ch.cancel();
  ASIO_CHECK(!ch.ready());

  ch.put(2, put_ec);
  ASIO_CHECK(put_ec == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(get_ec_1 == asio::error::operation_aborted);
  ASIO_CHECK(value_1 == 0);
  ASIO_CHECK(!ch.ready());

  // get 1 / close

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  ch.close();
  ASIO_CHECK(!ch.is_open());
  ASIO_CHECK(ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(get_ec_1 == asio::error::broken_pipe);
  ASIO_CHECK(value_1 == 0);
  ASIO_CHECK(ch.ready());

  ch.reset();
  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());

  // close / get 1

  ch.close();
  ASIO_CHECK(!ch.is_open());
  ASIO_CHECK(ch.ready());

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(get_ec_1 == asio::error::broken_pipe);
  ASIO_CHECK(value_1 == 0);
  ASIO_CHECK(ch.ready());

  ch.reset();
  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());
}

void int_buffer0_async_async_test()
{
  asio::io_service io_service;
  asio::channel<int> ch(io_service, 0);
  asio::error_code put_ec_1, put_ec_2, get_ec_1, get_ec_2;
  int value_1 = 0, value_2 = 0;

  ASIO_CHECK(ch.is_open());
  ASIO_CHECK(!ch.ready());

  // put 1 / get 1

  put_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_put(1, bindns::bind(&put_handler, _1, &put_ec_1));
  ASIO_CHECK(put_ec_1 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(!put_ec_1);
  ASIO_CHECK(!get_ec_1);
  ASIO_CHECK(value_1 == 1);

  // get 1 / put 1

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  put_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_put(1, bindns::bind(&put_handler, _1, &put_ec_1));
  ASIO_CHECK(put_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(!get_ec_1);
  ASIO_CHECK(value_1 == 1);
  ASIO_CHECK(!put_ec_1);

  // put 2 / get 2

  put_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_put(1, bindns::bind(&put_handler, _1, &put_ec_1));
  ASIO_CHECK(put_ec_1 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  put_ec_2 = asio::error::would_block;
  value_2 = 0;
  ch.async_put(2, bindns::bind(&put_handler, _1, &put_ec_2));
  ASIO_CHECK(put_ec_2 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  get_ec_2 = asio::error::would_block;
  value_2 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_2, &value_2));
  ASIO_CHECK(get_ec_2 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(!put_ec_1);
  ASIO_CHECK(!put_ec_2);
  ASIO_CHECK(!get_ec_1);
  ASIO_CHECK(value_1 == 1);
  ASIO_CHECK(!get_ec_2);
  ASIO_CHECK(value_2 == 2);

  // get 2 / put 2

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  get_ec_2 = asio::error::would_block;
  value_2 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_2, &value_2));
  ASIO_CHECK(get_ec_2 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  put_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_put(1, bindns::bind(&put_handler, _1, &put_ec_1));
  ASIO_CHECK(put_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  put_ec_2 = asio::error::would_block;
  value_2 = 0;
  ch.async_put(2, bindns::bind(&put_handler, _1, &put_ec_2));
  ASIO_CHECK(put_ec_2 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(!get_ec_1);
  ASIO_CHECK(value_1 == 1);
  ASIO_CHECK(!get_ec_2);
  ASIO_CHECK(value_2 == 2);
  ASIO_CHECK(!put_ec_1);
  ASIO_CHECK(!put_ec_2);

  // put 1 / get 2 / put 1

  put_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_put(1, bindns::bind(&put_handler, _1, &put_ec_1));
  ASIO_CHECK(put_ec_1 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  get_ec_2 = asio::error::would_block;
  value_2 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_2, &value_2));
  ASIO_CHECK(get_ec_2 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  put_ec_2 = asio::error::would_block;
  value_2 = 0;
  ch.async_put(2, bindns::bind(&put_handler, _1, &put_ec_2));
  ASIO_CHECK(put_ec_2 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(!put_ec_1);
  ASIO_CHECK(!get_ec_1);
  ASIO_CHECK(value_1 == 1);
  ASIO_CHECK(!get_ec_2);
  ASIO_CHECK(value_2 == 2);
  ASIO_CHECK(!put_ec_2);

  // get 1 / put 2 / get 1

  get_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_1, &value_1));
  ASIO_CHECK(get_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  put_ec_1 = asio::error::would_block;
  value_1 = 0;
  ch.async_put(1, bindns::bind(&put_handler, _1, &put_ec_1));
  ASIO_CHECK(put_ec_1 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  put_ec_2 = asio::error::would_block;
  value_2 = 0;
  ch.async_put(2, bindns::bind(&put_handler, _1, &put_ec_2));
  ASIO_CHECK(put_ec_2 == asio::error::would_block);
  ASIO_CHECK(ch.ready());

  get_ec_2 = asio::error::would_block;
  value_2 = 0;
  ch.async_get(bindns::bind(&get_handler, _1, _2, &get_ec_2, &value_2));
  ASIO_CHECK(get_ec_2 == asio::error::would_block);
  ASIO_CHECK(!ch.ready());

  io_service.reset();
  io_service.run();
  ASIO_CHECK(!get_ec_1);
  ASIO_CHECK(value_1 == 1);
  ASIO_CHECK(!put_ec_1);
  ASIO_CHECK(!put_ec_2);
  ASIO_CHECK(!get_ec_2);
  ASIO_CHECK(value_2 == 2);
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
#endif

} // namespace channel_test

std::size_t sizes[] = { 0, 1, 2, 3, 4 };

ASIO_TEST_SUITE
(
  "channel",
  ASIO_PARAM_TEST_CASE(
    channel_test::sync_put_sync_get_test, sizes, sizes + 5)
  ASIO_PARAM_TEST_CASE(
    channel_test::async_put_sync_get_test, sizes, sizes + 5)
)
