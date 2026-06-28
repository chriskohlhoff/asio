//
// channel.cpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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

#include <utility>
#include "asio/any_completion_handler.hpp"
#include "asio/bind_executor.hpp"
#include "asio/bind_immediate_executor.hpp"
#include "asio/error.hpp"
#include "asio/inline_executor.hpp"
#include "asio/io_context.hpp"
#include "unit_test.hpp"

using asio::inline_executor;
using asio::io_context;
using asio::channel;

void unbuffered_channel_test()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  ASIO_CHECK(ch1.is_open());
  ASIO_CHECK(!ch1.can_receive());

  bool b1 = ch1.try_send(asio::error::eof, "hello");

  ASIO_CHECK(!b1);

  std::string s1 = "abcdefghijklmnopqrstuvwxyz";
  bool b2 = ch1.try_send(asio::error::eof, std::move(s1));

  ASIO_CHECK(!b2);
  ASIO_CHECK(!s1.empty());

  asio::error_code ec1;
  std::string s2;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec1 = ec;
        s2 = std::move(s);
      });

  bool b3 = ch1.try_send(asio::error::eof, std::move(s1));

  ASIO_CHECK(b3);
  ASIO_CHECK(s1.empty());

  ctx.run();

  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s2 == "abcdefghijklmnopqrstuvwxyz");

  bool b4 = ch1.try_receive([](asio::error_code, std::string){});

  ASIO_CHECK(!b4);

  asio::error_code ec2 = asio::error::would_block;
  std::string s3 = "zyxwvutsrqponmlkjihgfedcba";
  ch1.async_send(asio::error::eof, std::move(s3),
      [&](asio::error_code ec)
      {
        ec2 = ec;
      });

  asio::error_code ec3;
  std::string s4;
  bool b5 = ch1.try_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec3 = ec;
        s4 = s;
      });

  ASIO_CHECK(b5);
  ASIO_CHECK(ec3 == asio::error::eof);
  ASIO_CHECK(s4 == "zyxwvutsrqponmlkjihgfedcba");

  ctx.restart();
  ctx.run();

  ASIO_CHECK(!ec2);
}

void buffered_channel_test()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx, 1);

  ASIO_CHECK(ch1.is_open());
  ASIO_CHECK(!ch1.can_receive());

  bool b1 = ch1.try_send(asio::error::eof, "hello");

  ASIO_CHECK(b1);

  std::string s1 = "abcdefghijklmnopqrstuvwxyz";
  bool b2 = ch1.try_send(asio::error::eof, std::move(s1));

  ASIO_CHECK(!b2);
  ASIO_CHECK(!s1.empty());

  asio::error_code ec1;
  std::string s2;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec1 = ec;
        s2 = std::move(s);
      });

  ctx.run();

  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s2 == "hello");

  bool b4 = ch1.try_receive([](asio::error_code, std::string){});

  ASIO_CHECK(!b4);

  asio::error_code ec2 = asio::error::would_block;
  std::string s3 = "zyxwvutsrqponmlkjihgfedcba";
  ch1.async_send(asio::error::eof, std::move(s3),
      [&](asio::error_code ec)
      {
        ec2 = ec;
      });

  asio::error_code ec3;
  std::string s4;
  bool b5 = ch1.try_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec3 = ec;
        s4 = s;
      });

  ASIO_CHECK(b5);
  ASIO_CHECK(ec3 == asio::error::eof);
  ASIO_CHECK(s4 == "zyxwvutsrqponmlkjihgfedcba");

  ctx.restart();
  ctx.run();

  ASIO_CHECK(!ec2);

  bool b6 = ch1.try_send(asio::error_code(), "goodbye");

  ASIO_CHECK(b6);

  ch1.close();

  asio::error_code ec4;
  std::string s5;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec4 = ec;
        s5 = std::move(s);
      });

  ctx.restart();
  ctx.run();

  ASIO_CHECK(!ec4);
  ASIO_CHECK(s5 == "goodbye");

  asio::error_code ec5;
  std::string s6;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec5 = ec;
        s6 = std::move(s);
      });

  ctx.restart();
  ctx.run();

  ASIO_CHECK(ec5 == asio::channel_errc::channel_closed);
  ASIO_CHECK(s6.empty());
}

void buffered_error_channel_test()
{
  io_context ctx;

  channel<void(asio::error_code)> ch1(ctx, 1);

  ASIO_CHECK(ch1.is_open());
  ASIO_CHECK(!ch1.can_receive());

  bool b1 = ch1.try_send(asio::error::eof);

  ASIO_CHECK(b1);

  bool b2 = ch1.try_send(asio::error::eof);

  ASIO_CHECK(!b2);

  asio::error_code ec1;
  ch1.async_receive(
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ctx.run();

  ASIO_CHECK(ec1 == asio::error::eof);

  bool b4 = ch1.try_receive([](asio::error_code){});

  ASIO_CHECK(!b4);

  asio::error_code ec2 = asio::error::would_block;
  ch1.async_send(asio::error::eof,
      [&](asio::error_code ec)
      {
        ec2 = ec;
      });

  asio::error_code ec3;
  bool b5 = ch1.try_receive(
      [&](asio::error_code ec)
      {
        ec3 = ec;
      });

  ASIO_CHECK(b5);
  ASIO_CHECK(ec3 == asio::error::eof);

  ctx.restart();
  ctx.run();

  ASIO_CHECK(!ec2);
}

void unbuffered_non_immediate_receive()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s1),
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec2 = ec;
        s2 = std::move(s);
      });

  ASIO_CHECK(ec2 == asio::error::would_block);

  ctx.run();

  ASIO_CHECK(!ec1);
  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "0123456789");
}

void unbuffered_immediate_receive()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s1),
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2;
  ch1.async_receive(
      bind_immediate_executor(inline_executor(),
        [&](asio::error_code ec, std::string s)
        {
          ec2 = ec;
          s2 = std::move(s);
        }));

  ASIO_CHECK(ec1 == asio::error::would_block);
  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "0123456789");

  ctx.run();

  ASIO_CHECK(!ec1);
}

void unbuffered_executor_receive()
{
  io_context ctx;
  io_context ctx2;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s1),
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2;
  ch1.async_receive(
      bind_executor(ctx2,
        [&](asio::error_code ec, std::string s)
        {
          ec2 = ec;
          s2 = std::move(s);
        }));

  ASIO_CHECK(ec1 == asio::error::would_block);
  ASIO_CHECK(ec2 == asio::error::would_block);

  ctx.run();

  ASIO_CHECK(!ec1);
  ASIO_CHECK(ec2 == asio::error::would_block);

  ctx2.run();

  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "0123456789");
}

void unbuffered_non_immediate_send()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec1 = ec;
        s1 = std::move(s);
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s2),
      [&](asio::error_code ec)
      {
        ec2 = ec;
      });

  ASIO_CHECK(ec2 == asio::error::would_block);

  ctx.run();

  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "0123456789");
  ASIO_CHECK(!ec2);
}

void unbuffered_immediate_send()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec1 = ec;
        s1 = std::move(s);
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s2),
      bind_immediate_executor(inline_executor(),
        [&](asio::error_code ec)
        {
          ec2 = ec;
        }));

  ASIO_CHECK(ec1 == asio::error::would_block);
  ASIO_CHECK(!ec2);

  ctx.run();

  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "0123456789");
}

void unbuffered_executor_send()
{
  io_context ctx;
  io_context ctx2;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec1 = ec;
        s1 = std::move(s);
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s2),
      bind_executor(ctx2,
        [&](asio::error_code ec)
        {
          ec2 = ec;
        }));

  ASIO_CHECK(ec1 == asio::error::would_block);
  ASIO_CHECK(ec2 == asio::error::would_block);

  ctx.run();

  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "0123456789");
  ASIO_CHECK(ec2 == asio::error::would_block);

  ctx2.run();

  ASIO_CHECK(!ec2);
}

void buffered_non_immediate_receive()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx, 1);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s1),
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  ctx.run();

  ASIO_CHECK(!ec1);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec2 = ec;
        s2 = std::move(s);
      });

  ASIO_CHECK(ec2 == asio::error::would_block);

  ctx.restart();
  ctx.run();

  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "0123456789");
}

void buffered_immediate_receive()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx, 1);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s1),
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  ctx.run();

  ASIO_CHECK(!ec1);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2;
  ch1.async_receive(
      bind_immediate_executor(inline_executor(),
        [&](asio::error_code ec, std::string s)
        {
          ec2 = ec;
          s2 = std::move(s);
        }));

  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "0123456789");

  ctx.restart();
  ctx.run();
}

void buffered_executor_receive()
{
  io_context ctx;
  io_context ctx2;

  channel<void(asio::error_code, std::string)> ch1(ctx, 1);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s1),
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  ctx.run();

  ASIO_CHECK(!ec1);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2;
  ch1.async_receive(
      bind_executor(ctx2,
        [&](asio::error_code ec, std::string s)
        {
          ec2 = ec;
          s2 = std::move(s);
        }));

  ASIO_CHECK(ec2 == asio::error::would_block);

  ctx.restart();
  ctx.run();

  ASIO_CHECK(ec2 == asio::error::would_block);

  ctx2.run();

  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "0123456789");
}

void buffered_non_immediate_send()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx, 1);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s1),
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  ctx.run();

  ASIO_CHECK(!ec1);
}

void buffered_immediate_send()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx, 1);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s1),
      bind_immediate_executor(inline_executor(),
        [&](asio::error_code ec)
        {
          ec1 = ec;
        }));

  ASIO_CHECK(!ec1);

  ctx.run();
}

void buffered_executor_send()
{
  io_context ctx;
  io_context ctx2;

  channel<void(asio::error_code, std::string)> ch1(ctx, 1);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1 = "0123456789";
  ch1.async_send(asio::error::eof, std::move(s1),
      bind_executor(ctx2,
        [&](asio::error_code ec)
        {
          ec1 = ec;
        }));

  ASIO_CHECK(ec1 == asio::error::would_block);

  ctx.run();

  ASIO_CHECK(ec1 == asio::error::would_block);

  ctx2.run();

  ASIO_CHECK(!ec1);
}

void try_send_via_dispatch()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1;
  ch1.async_receive(
      bind_executor(asio::inline_executor(),
        [&](asio::error_code ec, std::string s)
        {
          ec1 = ec;
          s1 = std::move(s);
        }));

  ASIO_CHECK(ec1 == asio::error::would_block);

  ctx.poll();

  ASIO_CHECK(ec1 == asio::error::would_block);

  std::string s2 = "0123456789";
  ch1.try_send_via_dispatch(asio::error::eof, std::move(s2));

  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "0123456789");
  ASIO_CHECK(s2.empty());
}

void try_send_n_via_dispatch()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1;
  ch1.async_receive(
      bind_executor(asio::inline_executor(),
        [&](asio::error_code ec, std::string s)
        {
          ec1 = ec;
          s1 = std::move(s);
        }));

  ASIO_CHECK(ec1 == asio::error::would_block);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2;
  ch1.async_receive(
      bind_executor(asio::inline_executor(),
        [&](asio::error_code ec, std::string s)
        {
          ec2 = ec;
          s2 = std::move(s);
        }));

  ASIO_CHECK(ec1 == asio::error::would_block);

  ctx.poll();

  ASIO_CHECK(ec1 == asio::error::would_block);
  ASIO_CHECK(ec2 == asio::error::would_block);

  std::string s3 = "0123456789";
  ch1.try_send_n_via_dispatch(2, asio::error::eof, std::move(s3));

  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "0123456789");
  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "0123456789");
  ASIO_CHECK(s3.empty());
}

struct counting_handler
{
  int& count_;
  template <typename... Args>
  void operator()(Args...) { ++count_; }
};

void try_peek_test()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx, 2);

  bool b1 = ch1.try_peek([](asio::error_code, std::string){});

  ASIO_CHECK(!b1);

  bool b2 = ch1.try_send(asio::error::eof, "hello");

  ASIO_CHECK(b2);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1;
  bool b3 = ch1.try_peek(
      [&](asio::error_code ec, std::string s)
      {
        ec1 = ec;
        s1 = s;
      });

  ASIO_CHECK(b3);
  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "hello");

  asio::error_code ec2 = asio::error::would_block;
  std::string s2;
  bool b4 = ch1.try_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec2 = ec;
        s2 = s;
      });

  ASIO_CHECK(b4);
  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "hello");

  bool b5 = ch1.try_peek([](asio::error_code, std::string){});

  ASIO_CHECK(!b5);

  asio::error_code ec3 = asio::error::would_block;
  std::string s3 = "zyxwvutsrqponmlkjihgfedcba";
  ch1.async_send(asio::error::eof, std::move(s3),
      [&](asio::error_code ec)
      {
        ec3 = ec;
      });

  asio::error_code ec4 = asio::error::would_block;
  std::string s4;
  bool b6 = ch1.try_peek(
      [&](asio::error_code ec, std::string s)
      {
        ec4 = ec;
        s4 = s;
      });

  ASIO_CHECK(b6);
  ASIO_CHECK(ec4 == asio::error::eof);
  ASIO_CHECK(s4 == "zyxwvutsrqponmlkjihgfedcba");

  ASIO_CHECK(ec3 == asio::error::would_block);

  asio::error_code ec5 = asio::error::would_block;
  std::string s5;
  bool b7 = ch1.try_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec5 = ec;
        s5 = s;
      });

  ASIO_CHECK(b7);
  ASIO_CHECK(ec5 == asio::error::eof);
  ASIO_CHECK(s5 == "zyxwvutsrqponmlkjihgfedcba");

  ctx.run();

  ASIO_CHECK(!ec3);

  channel<void()> ch2(ctx, 2);
  int zero = 0;

  bool b8 = ch2.try_peek(counting_handler{zero});

  ASIO_CHECK(!b8);
  ASIO_CHECK(zero == 0);

  bool b9 = ch2.try_send();

  ASIO_CHECK(b9);

  int peek_count = 0;
  bool b10 = ch2.try_peek(counting_handler{peek_count});

  ASIO_CHECK(b10);
  ASIO_CHECK(peek_count == 1);

  int receive_count = 0;
  bool b11 = ch2.try_receive(counting_handler{receive_count});

  ASIO_CHECK(b11);
  ASIO_CHECK(receive_count == 1);

  bool b12 = ch2.try_peek(counting_handler{zero});

  ASIO_CHECK(!b12);
  ASIO_CHECK(zero == 0);
}

struct multi_signature_handler
{
  std::string* s_;
  asio::error_code* ec_;

  void operator()(std::string s)
  {
    *s_ = s;
  }

  void operator()(asio::error_code ec)
  {
    *ec_ = ec;
  }
};

void implicit_error_signature_channel_test()
{
  io_context ctx;

  channel<void(std::string)> ch1(ctx);

  ASIO_CHECK(ch1.is_open());
  ASIO_CHECK(!ch1.can_receive());

  bool b1 = ch1.try_send("hello");

  ASIO_CHECK(!b1);

  std::string s1 = "abcdefghijklmnopqrstuvwxyz";
  bool b2 = ch1.try_send(std::move(s1));

  ASIO_CHECK(!b2);
  ASIO_CHECK(!s1.empty());

  std::string s2;
  asio::error_code ec1 = asio::error::would_block;
  multi_signature_handler h1 = {&s2, &ec1};
  ch1.async_receive(h1);

  bool b3 = ch1.try_send(std::move(s1));

  ASIO_CHECK(b3);
  ASIO_CHECK(s1.empty());

  ctx.run();

  ASIO_CHECK(s2 == "abcdefghijklmnopqrstuvwxyz");
  ASIO_CHECK(ec1 == asio::error::would_block);

  std::string s3;
  asio::error_code ec2;
  multi_signature_handler h2 = {&s3, &ec2};
  bool b4 = ch1.try_receive(h2);

  ASIO_CHECK(!b4);

  std::string s4 = "zyxwvutsrqponmlkjihgfedcba";
  asio::error_code ec3;
  ch1.async_send(std::move(s4),
      [&](asio::error_code ec)
      {
        ec3 = ec;
      });

  std::string s5;
  asio::error_code ec4 = asio::error::would_block;
  multi_signature_handler h3 = {&s5, &ec4};
  bool b5 = ch1.try_receive(h3);

  ASIO_CHECK(b5);
  ASIO_CHECK(ec4 == asio::error::would_block);
  ASIO_CHECK(s5 == "zyxwvutsrqponmlkjihgfedcba");

  ctx.restart();
  ctx.run();

  ASIO_CHECK(!ec3);

  std::string s6;
  asio::error_code ec5 = asio::error::would_block;
  multi_signature_handler h4 = {&s6, &ec5};
  ch1.async_receive(h4);

  ch1.close();

  ctx.restart();
  ctx.run();

  ASIO_CHECK(s6.empty());
  ASIO_CHECK(ec5 == asio::channel_errc::channel_closed);
}

void channel_with_any_completion_handler_test()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  std::string s1;
  ch1.async_receive(
      asio::any_completion_handler<
        void(asio::error_code, std::string)>(
          [&](asio::error_code ec, std::string s)
          {
            ec1 = ec;
            s1 = std::move(s);
          }));

  asio::error_code ec2 = asio::error::would_block;
  std::string s2 = "zyxwvutsrqponmlkjihgfedcba";
  ch1.async_send(asio::error::eof, std::move(s2),
      asio::any_completion_handler<void(asio::error_code)>(
        [&](asio::error_code ec)
        {
          ec2 = ec;
        }));

  ASIO_CHECK(ec1 == asio::error::would_block);
  ASIO_CHECK(ec2 == asio::error::would_block);

  ctx.run();

  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "zyxwvutsrqponmlkjihgfedcba");
  ASIO_CHECK(!ec2);
}

void can_send_can_receive_test()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  ASIO_CHECK(!ch1.can_send());
  ASIO_CHECK(!ch1.can_receive());

  asio::error_code ec1 = asio::error::would_block;
  std::string s1 = "hello";
  ch1.async_send(asio::error::eof, std::move(s1),
      [&](asio::error_code ec) { ec1 = ec; });

  ASIO_CHECK(!ch1.can_send());
  ASIO_CHECK(ch1.can_receive());

  bool b1 = ch1.try_receive([](asio::error_code, std::string){});

  ASIO_CHECK(b1);
  ASIO_CHECK(!ch1.can_send());
  ASIO_CHECK(!ch1.can_receive());

  ctx.run();

  ASIO_CHECK(!ec1);

  asio::error_code ec2 = asio::error::would_block;
  std::string s2;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec2 = ec;
        s2 = std::move(s);
      });

  ASIO_CHECK(ch1.can_send());
  ASIO_CHECK(!ch1.can_receive());

  bool b2 = ch1.try_send(asio::error::eof, "world");

  ASIO_CHECK(b2);
  ASIO_CHECK(!ch1.can_send());
  ASIO_CHECK(!ch1.can_receive());

  ctx.restart();
  ctx.run();

  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "world");

  channel<void(asio::error_code, std::string)> ch2(ctx, 2);

  ASIO_CHECK(ch2.can_send());
  ASIO_CHECK(!ch2.can_receive());

  ASIO_CHECK(ch2.try_send(asio::error_code(), "first"));

  ASIO_CHECK(ch2.can_send());
  ASIO_CHECK(ch2.can_receive());

  ASIO_CHECK(ch2.try_send(asio::error_code(), "second"));

  ASIO_CHECK(!ch2.can_send());
  ASIO_CHECK(ch2.can_receive());

  asio::error_code ec3;
  std::string s3;
  ASIO_CHECK(ch2.try_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec3 = ec;
        s3 = std::move(s);
      }));

  ASIO_CHECK(!ec3);
  ASIO_CHECK(s3 == "first");
  ASIO_CHECK(ch2.can_send());
  ASIO_CHECK(ch2.can_receive());
}

void unbuffered_cancel_after_close()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  ch1.async_send(asio::error::eof, "hello",
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  ch1.close();
  ch1.cancel();

  ctx.run();

  ASIO_CHECK(ec1 == asio::channel_errc::channel_cancelled);
}

void buffered_cancel_after_close()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx, 1);

  bool b1 = ch1.try_send(asio::error::eof, "first");

  ASIO_CHECK(b1);

  asio::error_code ec1 = asio::error::would_block;
  ch1.async_send(asio::error::eof, "second",
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  ch1.close();
  ch1.cancel();

  ctx.run();

  ASIO_CHECK(ec1 == asio::channel_errc::channel_cancelled);
}

void unbuffered_destroy_after_close()
{
  io_context ctx;

  asio::error_code ec1 = asio::error::would_block;

  {
    channel<void(asio::error_code, std::string)> ch1(ctx);

    ch1.async_send(asio::error::eof, "hello",
        [&](asio::error_code ec)
        {
          ec1 = ec;
        });

    ASIO_CHECK(ec1 == asio::error::would_block);

    ch1.close();
  }

  ctx.run();

  ASIO_CHECK(ec1 == asio::channel_errc::channel_cancelled);
}

void unbuffered_drain_after_close()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  ch1.async_send(asio::error::eof, "first",
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  asio::error_code ec2 = asio::error::would_block;
  ch1.async_send(asio::error::eof, "second",
      [&](asio::error_code ec)
      {
        ec2 = ec;
      });

  ch1.close();

  ASIO_CHECK(!ch1.is_open());
  ASIO_CHECK(ch1.ready());

  asio::error_code ec3;
  std::string s3;
  bool b1 = ch1.try_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec3 = ec;
        s3 = std::move(s);
      });

  ASIO_CHECK(b1);
  ASIO_CHECK(ec3 == asio::error::eof);
  ASIO_CHECK(s3 == "first");

  asio::error_code ec4 = asio::error::would_block;
  std::string s4;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec4 = ec;
        s4 = std::move(s);
      });

  asio::error_code ec5 = asio::error::would_block;
  std::string s5;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec5 = ec;
        s5 = std::move(s);
      });

  ctx.run();

  ASIO_CHECK(!ec1);
  ASIO_CHECK(!ec2);
  ASIO_CHECK(ec4 == asio::error::eof);
  ASIO_CHECK(s4 == "second");
  ASIO_CHECK(ec5 == asio::channel_errc::channel_closed);
  ASIO_CHECK(s5.empty());
}

void buffered_drain_after_close()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx, 1);

  bool b1 = ch1.try_send(asio::error::eof, "first");

  ASIO_CHECK(b1);

  asio::error_code ec1 = asio::error::would_block;
  ch1.async_send(asio::error::eof, "second",
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ASIO_CHECK(ec1 == asio::error::would_block);

  ch1.close();

  ASIO_CHECK(!ch1.is_open());
  ASIO_CHECK(ch1.ready());

  asio::error_code ec2;
  std::string s2;
  bool b2 = ch1.try_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec2 = ec;
        s2 = std::move(s);
      });

  ASIO_CHECK(b2);
  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "first");

  asio::error_code ec3 = asio::error::would_block;
  std::string s3;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec3 = ec;
        s3 = std::move(s);
      });

  asio::error_code ec4 = asio::error::would_block;
  std::string s4;
  ch1.async_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec4 = ec;
        s4 = std::move(s);
      });

  ctx.run();

  ASIO_CHECK(!ec1);
  ASIO_CHECK(ec3 == asio::error::eof);
  ASIO_CHECK(s3 == "second");
  ASIO_CHECK(ec4 == asio::channel_errc::channel_closed);
  ASIO_CHECK(s4.empty());
}

void send_after_close()
{
  io_context ctx;

  channel<void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  ch1.async_send(asio::error::eof, "hello",
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  ch1.close();

  bool b1 = ch1.try_send(asio::error::eof, "rejected");

  ASIO_CHECK(!b1);

  asio::error_code ec2 = asio::error::would_block;
  ch1.async_send(asio::error::eof, "rejected",
      [&](asio::error_code ec)
      {
        ec2 = ec;
      });

  asio::error_code ec3;
  std::string s3;
  bool b2 = ch1.try_receive(
      [&](asio::error_code ec, std::string s)
      {
        ec3 = ec;
        s3 = std::move(s);
      });

  ASIO_CHECK(b2);
  ASIO_CHECK(ec3 == asio::error::eof);
  ASIO_CHECK(s3 == "hello");

  ctx.run();

  ASIO_CHECK(!ec1);
  ASIO_CHECK(ec2 == asio::channel_errc::channel_closed);
}

struct int_or_string_handler
{
  asio::error_code* ec_;
  int* i_;
  std::string* s_;

  void operator()(asio::error_code ec, int i)
  {
    *ec_ = ec;
    *i_ = i;
  }

  void operator()(asio::error_code ec, std::string s)
  {
    *ec_ = ec;
    *s_ = std::move(s);
  }
};

struct three_signature_handler
{
  asio::error_code* ec_;
  int* i_;
  std::string* s1_;
  std::string* s2_;

  void operator()(asio::error_code ec, int i)
  {
    *ec_ = ec;
    *i_ = i;
  }

  void operator()(asio::error_code ec, std::string s)
  {
    *ec_ = ec;
    *s1_ = std::move(s);
  }

  void operator()(asio::error_code ec, std::string s1, std::string s2)
  {
    *ec_ = ec;
    *s1_ = std::move(s1);
    *s2_ = std::move(s2);
  }
};

void multi_signature_try_send()
{
  io_context ctx;

  channel<void(asio::error_code, int),
      void(asio::error_code, std::string)> ch1(ctx, 2);

  bool b1 = ch1.try_send(asio::error_code(), 42);
  bool b2 = ch1.try_send(asio::error::eof, std::string("second"));

  ASIO_CHECK(b1);
  ASIO_CHECK(b2);

  asio::error_code ec1;
  int i1 = 0;
  std::string s1;
  int_or_string_handler h1 = {&ec1, &i1, &s1};

  bool b3 = ch1.try_receive(h1);

  ASIO_CHECK(b3);
  ASIO_CHECK(!ec1);
  ASIO_CHECK(i1 == 42);

  bool b4 = ch1.try_receive(h1);

  ASIO_CHECK(b4);
  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "second");

  bool b5 = ch1.try_send(asio::error_code(), 0);
  bool b6 = ch1.try_send(asio::error_code(), 0);

  ASIO_CHECK(b5);
  ASIO_CHECK(b6);

  bool b7 = ch1.try_send(asio::error::eof, std::string("full"));

  ASIO_CHECK(!b7);

#if defined(ASIO_HAS_STD_VARIANT)
  channel<void(asio::error_code, int),
      void(asio::error_code, std::string),
      void(asio::error_code, std::string, std::string)> ch2(ctx, 3);

  bool b8 = ch2.try_send(asio::error_code(), 42);
  bool b9 = ch2.try_send(asio::error::eof, std::string("second"));
  bool b10 = ch2.try_send(asio::error::eof,
      std::string("left"), std::string("right"));

  ASIO_CHECK(b8);
  ASIO_CHECK(b9);
  ASIO_CHECK(b10);

  asio::error_code ec2;
  int i2 = 0;
  std::string s2;
  std::string s3;
  three_signature_handler h2 = {&ec2, &i2, &s2, &s3};

  bool b11 = ch2.try_receive(h2);

  ASIO_CHECK(b11);
  ASIO_CHECK(!ec2);
  ASIO_CHECK(i2 == 42);

  bool b12 = ch2.try_receive(h2);

  ASIO_CHECK(b12);
  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "second");

  bool b13 = ch2.try_receive(h2);

  ASIO_CHECK(b13);
  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s2 == "left");
  ASIO_CHECK(s3 == "right");

  bool b14 = ch2.try_send(asio::error::eof,
      std::string("buffer"), std::string("full"));
  bool b15 = ch2.try_send(asio::error::eof,
      std::string("buffer"), std::string("full"));
  bool b16 = ch2.try_send(asio::error::eof,
      std::string("buffer"), std::string("full"));
  bool b17 = ch2.try_send(asio::error::eof,
      std::string("buffer"), std::string("full"));

  ASIO_CHECK(b14);
  ASIO_CHECK(b15);
  ASIO_CHECK(b16);
  ASIO_CHECK(!b17);
#endif // defined(ASIO_HAS_STD_VARIANT)
}

struct long_or_int_handler
{
  int* which_;

  void operator()(asio::error_code, long)
  {
    *which_ = 1;
  }

  void operator()(asio::error_code, int)
  {
    *which_ = 2;
  }
};

void multi_signature_try_send_first_match_wins()
{
  io_context ctx;

  channel<void(asio::error_code, long),
      void(asio::error_code, int)> ch1(ctx, 1);

  bool b1 = ch1.try_send(asio::error_code(), 42);

  ASIO_CHECK(b1);

  int which = 0;
  long_or_int_handler h1 = {&which};
  bool b2 = ch1.try_receive(h1);

  ASIO_CHECK(b2);
  ASIO_CHECK(which == 1);
}

struct movable_payload
{
  int* moves_;
  int* copies_;
  int value_;

  movable_payload(int* moves, int* copies, int value)
    : moves_(moves),
      copies_(copies),
      value_(value)
  {
  }

  movable_payload(const movable_payload& other)
    : moves_(other.moves_),
      copies_(other.copies_),
      value_(other.value_)
  {
    ++*copies_;
  }

  movable_payload(movable_payload&& other)
    : moves_(other.moves_),
      copies_(other.copies_),
      value_(other.value_)
  {
    ++*moves_;
  }
};

void multi_signature_try_send_failure_preserves_arguments()
{
  io_context ctx;

  channel<void(asio::error_code, int),
      void(asio::error_code, movable_payload)> ch1(ctx);

  int moves = 0;
  int copies = 0;
  movable_payload p1(&moves, &copies, 42);

  bool b1 = ch1.try_send(asio::error::eof, std::move(p1));

  ASIO_CHECK(!b1);
  ASIO_CHECK(moves == 0);
  ASIO_CHECK(copies == 0);

  channel<void(asio::error_code, int),
      void(asio::error_code, movable_payload)> ch2(ctx, 1);

  bool b2 = ch2.try_send(asio::error::eof, std::move(p1));

  ASIO_CHECK(b2);
  ASIO_CHECK(moves > 0);
  ASIO_CHECK(copies == 0);
}

void multi_signature_try_send_n()
{
#if defined(ASIO_HAS_STD_VARIANT)
  io_context ctx;

  channel<void(asio::error_code, int),
      void(asio::error_code, std::string)> ch1(ctx, 4);

  std::size_t n1 = ch1.try_send_n(2, asio::error::eof, std::string("copied"));

  ASIO_CHECK(n1 == 2);

  asio::error_code ec1;
  int i1 = 0;
  std::string s1;
  int_or_string_handler h1 = {&ec1, &i1, &s1};

  bool b1 = ch1.try_receive(h1);

  ASIO_CHECK(b1);
  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "copied");

  s1.clear();
  bool b2 = ch1.try_receive(h1);

  ASIO_CHECK(b2);
  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "copied");
#endif // defined(ASIO_HAS_STD_VARIANT)
}

void multi_signature_try_send_via_dispatch()
{
  io_context ctx;

  channel<void(asio::error_code, int),
      void(asio::error_code, std::string)> ch1(ctx);

  asio::error_code ec1 = asio::error::would_block;
  int i1 = 0;
  std::string s1;
  int_or_string_handler h1 = {&ec1, &i1, &s1};
  ch1.async_receive(bind_executor(asio::inline_executor(), h1));

  ASIO_CHECK(ec1 == asio::error::would_block);

  ctx.poll();

  ASIO_CHECK(ec1 == asio::error::would_block);

  std::string s2 = "0123456789";
  bool b1 = ch1.try_send_via_dispatch(asio::error::eof, std::move(s2));

  ASIO_CHECK(b1);
  ASIO_CHECK(ec1 == asio::error::eof);
  ASIO_CHECK(s1 == "0123456789");
  ASIO_CHECK(s2.empty());

#if defined(ASIO_HAS_STD_VARIANT)
  asio::error_code ec2 = asio::error::would_block;
  int i2 = 0;
  std::string s3;
  int_or_string_handler h2 = {&ec2, &i2, &s3};
  ch1.async_receive(bind_executor(asio::inline_executor(), h2));

  asio::error_code ec3 = asio::error::would_block;
  int i3 = 0;
  std::string s4;
  int_or_string_handler h3 = {&ec3, &i3, &s4};
  ch1.async_receive(bind_executor(asio::inline_executor(), h3));

  ctx.poll();

  ASIO_CHECK(ec2 == asio::error::would_block);
  ASIO_CHECK(ec3 == asio::error::would_block);

  std::string s5 = "9876543210";
  std::size_t n1 = ch1.try_send_n_via_dispatch(
      2, asio::error::eof, std::move(s5));

  ASIO_CHECK(n1 == 2);
  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s3 == "9876543210");
  ASIO_CHECK(ec3 == asio::error::eof);
  ASIO_CHECK(s4 == "9876543210");
  ASIO_CHECK(s5.empty());
#endif // defined(ASIO_HAS_STD_VARIANT)
}

void multi_signature_async_send()
{
  io_context ctx;

  channel<void(asio::error_code, int),
      void(asio::error_code, std::string)> ch1(ctx, 1);

  asio::error_code ec1 = asio::error::would_block;
  ch1.async_send(asio::error::eof, std::string("hello"),
      [&](asio::error_code ec)
      {
        ec1 = ec;
      });

  asio::error_code ec2;
  int i1 = 0;
  std::string s1;
  int_or_string_handler h1 = {&ec2, &i1, &s1};
  bool b1 = ch1.try_receive(h1);

  ASIO_CHECK(b1);
  ASIO_CHECK(ec2 == asio::error::eof);
  ASIO_CHECK(s1 == "hello");

  ctx.run();

  ASIO_CHECK(!ec1);
}

void channel_move_test()
{
  io_context ctx;

  channel<void(asio::error_code)> ch1(ctx);
  channel<void(asio::error_code)> ch2 = std::move(ch1);
  (void)ch2;

  channel<void(asio::error_code, std::string)> ch3(ctx);
  channel<void(asio::error_code, std::string)> ch4 = std::move(ch3);
  (void)ch4;
}

ASIO_TEST_SUITE
(
  "channel",
  ASIO_TEST_CASE(unbuffered_channel_test)
  ASIO_TEST_CASE(buffered_channel_test)
  ASIO_TEST_CASE(buffered_error_channel_test)
  ASIO_TEST_CASE(unbuffered_non_immediate_receive)
  ASIO_TEST_CASE(unbuffered_immediate_receive)
  ASIO_TEST_CASE(unbuffered_executor_receive)
  ASIO_TEST_CASE(unbuffered_non_immediate_send)
  ASIO_TEST_CASE(unbuffered_immediate_send)
  ASIO_TEST_CASE(unbuffered_executor_send)
  ASIO_TEST_CASE(buffered_non_immediate_receive)
  ASIO_TEST_CASE(buffered_immediate_receive)
  ASIO_TEST_CASE(buffered_executor_receive)
  ASIO_TEST_CASE(buffered_non_immediate_send)
  ASIO_TEST_CASE(buffered_immediate_send)
  ASIO_TEST_CASE(buffered_executor_send)
  ASIO_TEST_CASE(try_send_via_dispatch)
  ASIO_TEST_CASE(try_send_n_via_dispatch)
  ASIO_TEST_CASE(try_peek_test)
  ASIO_TEST_CASE(can_send_can_receive_test)
  ASIO_TEST_CASE(implicit_error_signature_channel_test)
  ASIO_TEST_CASE(channel_with_any_completion_handler_test)
  ASIO_TEST_CASE(unbuffered_cancel_after_close)
  ASIO_TEST_CASE(buffered_cancel_after_close)
  ASIO_TEST_CASE(unbuffered_destroy_after_close)
  ASIO_TEST_CASE(unbuffered_drain_after_close)
  ASIO_TEST_CASE(buffered_drain_after_close)
  ASIO_TEST_CASE(send_after_close)
  ASIO_COMPILE_TEST_CASE(channel_move_test)
  ASIO_TEST_CASE(multi_signature_try_send)
  ASIO_TEST_CASE(multi_signature_try_send_first_match_wins)
  ASIO_TEST_CASE(multi_signature_try_send_failure_preserves_arguments)
  ASIO_TEST_CASE(multi_signature_try_send_n)
  ASIO_TEST_CASE(multi_signature_try_send_via_dispatch)
  ASIO_TEST_CASE(multi_signature_async_send)
)
