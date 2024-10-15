//
// process.cpp
// ~~~~~~~~
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
#include "asio/environment.hpp"
#include "asio/process.hpp"
#include "asio/process/stdio.hpp"
#include "asio/process/environment.hpp"
#include "asio/process/start_dir.hpp"
#include "asio/connect_pipe.hpp"
#include "asio/read_until.hpp"
#include "asio/read.hpp"
#include "asio/readable_pipe.hpp"
#include "asio/streambuf.hpp"
#include "asio/writable_pipe.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>

namespace asio
{

template struct basic_process<any_io_executor>;
}

#if defined(ASIO_WINDOWS)
#include "asio/process/windows/creation_flags.hpp"
#include "asio/process/windows/show_window.hpp"
#endif

#include "unit_test.hpp"

namespace process
{

asio::filesystem::path myself;

#if defined(ASIO_WINDOWS)
asio::filesystem::path shell()
{
  return asio::environment::find_executable("cmd");
}

asio::filesystem::path closable()
{
  return asio::environment::find_executable("notepad");
}

asio::filesystem::path interruptable()
{
  return asio::environment::find_executable("cmd");
}
#else
asio::filesystem::path shell()
{
  return asio::environment::find_executable("sh");
}
asio::filesystem::path closable()
{
  return asio::environment::find_executable("tee");
}
asio::filesystem::path interruptable()
{
  return asio::environment::find_executable("tee");
}
#endif

inline void trim_end(std::string & str)
{
  auto itr = std::find_if(str.rbegin(), str.rend(), [](char c) {return !std::isspace(c);});
  str.erase(itr.base(), str.end());
}

void terminate()
{
  asio::io_context ctx;

  auto sh = shell();
  ASIO_CHECK_MESSAGE(!sh.empty(), sh);
  asio::process proc(ctx, sh, {});
  proc.terminate();
  proc.wait();
}

void request_exit()
{
  asio::io_context ctx;

  auto sh = closable();
  ASIO_CHECK_MESSAGE(!sh.empty(), sh);
  asio::process proc(ctx, sh, {}
#if defined(ASIO_WINDOWS)
    , asio::windows::show_window_minimized_not_active
#endif
    );
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  proc.request_exit();
  proc.wait();
}

void interrupt()
{
  asio::io_context ctx;

  auto sh = interruptable();
  ASIO_CHECK_MESSAGE(!sh.empty(), sh);
  asio::process proc(ctx, sh, {}
#if defined(ASIO_WINDOWS)
  , asio::windows::create_new_process_group
#endif
  );
  proc.interrupt();
  proc.wait();
}

void print_args_out()
{
  asio::io_context ctx;

  asio::readable_pipe rp{ctx};
  asio::writable_pipe wp{ctx};
  asio::connect_pipe(rp, wp);

  asio::process proc(ctx, myself, {"--print-args", "foo", "bar"}, asio::process_stdio{.out=wp, .err=nullptr});

  wp.close();
  asio::streambuf st;
  std::istream is{&st};
  asio::error_code ec;

  auto sz = asio::read(rp, st,  ec);

  auto trim_end =
      [](std::string & str)
      {
        auto itr = std::find_if(str.rbegin(), str.rend(), [](char c) {return !std::isspace(c);});
        str.erase(itr.base(), str.end());
      };

  ASIO_CHECK(sz != 0);
  ASIO_CHECK_MESSAGE((ec == asio::error::broken_pipe) || (ec == asio::error::eof), ec.message());

  std::string line;
  ASIO_CHECK(std::getline(is, line));
  trim_end(line);
  ASIO_CHECK_MESSAGE(line == myself, line );

  ASIO_CHECK(std::getline(is, line));
  trim_end(line);
  ASIO_CHECK_MESSAGE(line == "--print-args", line);

  ASIO_CHECK(std::getline(is, line));
  trim_end(line);
  ASIO_CHECK_MESSAGE(line == "foo", line);

  ASIO_CHECK(std::getline(is, line));
  trim_end(line);
  ASIO_CHECK_MESSAGE(line == "bar", line);


  proc.wait();
  ASIO_CHECK(proc.exit_code() == 0);
}

void print_args_err()
{
  asio::io_context ctx;

  asio::readable_pipe rp{ctx};
  asio::writable_pipe wp{ctx};
  asio::connect_pipe(rp, wp);

  asio::process proc(ctx, myself, {"--print-args", "bar", "foo"}, asio::process_stdio{.out=nullptr, .err=wp});

  wp.close();
  asio::streambuf st;
  std::istream is{&st};
  asio::error_code ec;

  auto sz = asio::read(rp, st,  ec);



  ASIO_CHECK(sz != 0);
  ASIO_CHECK_MESSAGE((ec == asio::error::broken_pipe) || (ec == asio::error::eof), ec.message());

  std::string line;
  ASIO_CHECK(std::getline(is, line));
  trim_end(line);
  ASIO_CHECK_MESSAGE(line == myself, line );

  ASIO_CHECK(std::getline(is, line));
  trim_end(line);
  ASIO_CHECK_MESSAGE(line == "--print-args", line);

  ASIO_CHECK(std::getline(is, line));
  trim_end(line);
  ASIO_CHECK_MESSAGE(line == "bar", line);

  ASIO_CHECK(std::getline(is, line));
  trim_end(line);
  ASIO_CHECK_MESSAGE(line == "foo", line);


  proc.wait();
  ASIO_CHECK(proc.exit_code() == 0);
}

int print_args_impl(int argc, char * argv[])
{
  for (auto i = 0; i < argc; i++)
  {
    std::cout << argv[i] << std::endl;
    std::cerr << argv[i] << std::endl;
    if (!std::cout || !std::cerr)
      return 1;

  }
  return 0;
}

void echo_file()
{
  asio::io_context ctx;

  asio::readable_pipe rp{ctx};
  asio::writable_pipe wp{ctx};
  asio::connect_pipe(rp, wp);

  auto p = asio::filesystem::temp_directory_path() / "asio-test-thingy.txt";

  std::string test_data = "some ~~ test ~~ data";
  {
    std::ofstream ofs{p.string()};
    ofs.write(test_data.data(), test_data.size());
    ASIO_CHECK(ofs);
  }

  asio::process proc(ctx, myself, {"--echo"}, asio::process_stdio{.in=p, .out=wp});
  wp.close();

  std::string out;
  asio::error_code ec;

  auto sz = asio::read(rp, asio::dynamic_buffer(out),  ec);
  ASIO_CHECK(sz != 0);
  ASIO_CHECK_MESSAGE((ec == asio::error::broken_pipe) || (ec == asio::error::eof), ec.message());
  ASIO_CHECK_MESSAGE(out == test_data, out);

  proc.wait();
  ASIO_CHECK_MESSAGE(proc.exit_code() == 0, proc.exit_code());
}

void print_same_cwd()
{
  asio::io_context ctx;

  asio::readable_pipe rp{ctx};
  asio::writable_pipe wp{ctx};
  asio::connect_pipe(rp, wp);


  // default CWD
  asio::process proc(ctx, myself, {"--print-cwd"}, asio::process_stdio{.out=wp});
  wp.close();

  std::string out;
  asio::error_code ec;

  auto sz = asio::read(rp, asio::dynamic_buffer(out),  ec);
  ASIO_CHECK(sz != 0);
  ASIO_CHECK_MESSAGE((ec == asio::error::broken_pipe) || (ec == asio::error::eof), ec.message());
  ASIO_CHECK_MESSAGE(asio::filesystem::path(out) == asio::filesystem::current_path(),
                     asio::filesystem::path(out) << " != " << asio::filesystem::current_path());

  proc.wait();
  ASIO_CHECK_MESSAGE(proc.exit_code() == 0, proc.exit_code());
}

void print_other_cwd()
{
  asio::io_context ctx;

  asio::readable_pipe rp{ctx};
  asio::writable_pipe wp{ctx};
  asio::connect_pipe(rp, wp);

  auto tmp = asio::filesystem::canonical(asio::filesystem::temp_directory_path());

  // default CWD
  asio::process proc(ctx, myself, {"--print-cwd"}, asio::process_stdio{.out=wp}, asio::process_start_dir(tmp));
  wp.close();

  std::string out;
  asio::error_code ec;

  auto sz = asio::read(rp, asio::dynamic_buffer(out),  ec);
  ASIO_CHECK(sz != 0);
  ASIO_CHECK_MESSAGE((ec == asio::error::broken_pipe) || (ec == asio::error::eof), ec.message());
  ASIO_CHECK_MESSAGE(asio::filesystem::path(out) == tmp,
                     asio::filesystem::path(out) << " != " << tmp);

  proc.wait();
  ASIO_CHECK_MESSAGE(proc.exit_code() == 0, proc.exit_code() << " from " << proc.native_exit_code());
}



int echo_impl()
{
  std::cout << std::cin.rdbuf();
  return 0;
}

int print_cwd_impl()
{
  std::cout << asio::filesystem::current_path().string() << std::flush;
  return 0;
}

void check_eof()
{
  asio::io_context ctx;
  asio::process proc(ctx, myself, {"--check-eof"}, asio::process_stdio{.in=nullptr});
  proc.wait();
  ASIO_CHECK_MESSAGE(proc.exit_code() == 0, proc.exit_code());
}

int check_eof_impl()
{
  std::string st;
  std::cin >> st;
  return std::cin.eof() ? 0 : 1;
}

void exit_codes()
{
  asio::io_context ctx;
  asio::process proc(ctx, myself, {"--exit-code", "42"});
  proc.wait();
  ASIO_CHECK_MESSAGE(proc.exit_code() == 42, proc.exit_code());


  proc = asio::process(ctx, myself, {"--exit-code", "43"});

  bool done = false;
  int exit_code = 0;
  proc.async_wait([&](asio::error_code ec, int res)
                  {
                      done = true;
                      exit_code = asio::evaluate_exit_code(res);
                  });

  ctx.run();

  ASIO_CHECK_MESSAGE(exit_code == 43, exit_code);
  ASIO_CHECK(done);
}

template<typename ... Inits>
std::string read_env(const char * name, Inits && ... inits)
{
  asio::io_context ctx;

  asio::readable_pipe rp{ctx};
  asio::writable_pipe wp{ctx};
  asio::connect_pipe(rp, wp);

  asio::process proc(ctx, myself, {"--print-env", name}, asio::process_stdio{.out{wp}}, std::forward<Inits>(inits)...);

  wp.close();


  std::string out;
  asio::error_code ec;

  auto sz = asio::read(rp, asio::dynamic_buffer(out),  ec);
  ASIO_CHECK_MESSAGE((ec == asio::error::broken_pipe) || (ec == asio::error::eof), ec.message());

  trim_end(out);

  proc.wait();
  ASIO_CHECK(proc.exit_code() == 0);

  return out;
}

int print_env_impl(const char * c)
{
  auto e = ::getenv(c);
  if (e != nullptr)
    std::cout << e;
  return 0;
}

void environment()
{
  ASIO_CHECK(read_env("PATH") == ::getenv("PATH"));

  ASIO_CHECK(read_env("~~  surely does not exist ~~").empty());
  ASIO_CHECK("FOO-BAR" == read_env("FOOBAR", asio::process_environment{"FOOBAR=FOO-BAR"}));
  ASIO_CHECK("BAR-FOO" == read_env("PATH", asio::process_environment{"PATH=BAR-FOO", "XYZ=ZYX"}));
  ASIO_CHECK("BAR-FOO" == read_env("PATH", asio::process_environment{"PATH=BAR-FOO", "XYZ=ZYX"}));

#if defined(ASIO_WINDOWS)
  ASIO_CHECK("BAR-FOO" == read_env("PATH", asio::process_environment{L"PATH=BAR-FOO", L"XYZ=ZYX"}));
  ASIO_CHECK("BAR-FOO" == read_env("PATH", asio::process_environment{L"PATH=BAR-FOO", L"XYZ=ZYX"}));
  ASIO_CHECK("FOO-BAR" == read_env("FOOBAR", asio::process_environment{L"FOOBAR=FOO-BAR"}));
#endif

  ASIO_CHECK(read_env("PATH", asio::process_environment(asio::environment::view())) == ::getenv("PATH"));
}

}

int main(int argc, char * argv[])
try
{
  assert(argc > 0);
  process::myself = argv[0];
  if (argc > 1)
  {
    using std::operator""s;
    if (argv[1] == "--print-args"s)
      return process::print_args_impl(argc, argv);
    else if (argv[1] == "--echo"s)
      return process::echo_impl();
    else if (argv[1] == "--print-cwd"s)
      return process::print_cwd_impl();
    else if (argv[1] == "--check-eof"s)
      return process::check_eof_impl();
    else if (argv[1] == "--exit-code"s)
      return std::stoi(argv[2]);
    else if (argv[1] == "--print-env"s)
      return process::print_env_impl(argv[2]);
    else
      return 2;
  }
  asio::detail::begin_test_suite("process");

  ASIO_TEST_CASE(process::terminate);
  ASIO_TEST_CASE(process::request_exit);
  ASIO_TEST_CASE(process::interrupt);
  ASIO_TEST_CASE(process::print_args_out);
  ASIO_TEST_CASE(process::print_args_err);
  ASIO_TEST_CASE(process::echo_file);
  ASIO_TEST_CASE(process::print_other_cwd);
  ASIO_TEST_CASE(process::print_same_cwd);
  ASIO_TEST_CASE(process::check_eof);
  ASIO_TEST_CASE(process::exit_codes);
  ASIO_TEST_CASE(process::environment);

  return asio::detail::end_test_suite("process");

}
catch(std::exception & e)
{
  std::cerr << "Tests end with exception: " << e.what() << std::endl;
  return 1;
}