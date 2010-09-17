//
// udp_client.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio/ip/udp.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <algorithm>
#include <cstdio>
#include <vector>
#include "high_res_clock.hpp"

using asio::ip::udp;
using boost::posix_time::ptime;
using boost::posix_time::microsec_clock;

const int num_samples = 100000;

int main(int argc, char* argv[])
{
  if (argc != 5)
  {
    std::fprintf(stderr, "Usage: udp_client <ip> <port1> <nports> <bufsize>\n");
    return 1;
  }

  asio::io_service io_service;

  udp::socket socket(io_service, udp::endpoint(udp::v4(), 0));
  udp::socket::non_blocking_io nbio(true);
  socket.io_control(nbio);

  unsigned short first_port = std::atoi(argv[2]);
  unsigned short last_port = first_port + std::atoi(argv[3]) - 1;
  udp::endpoint target(asio::ip::address::from_string(argv[1]), first_port);

  std::vector<unsigned char> write_buf(std::atoi(argv[4]));
  std::vector<unsigned char> read_buf(std::atoi(argv[4]));

  ptime start = microsec_clock::universal_time();
  boost::uint64_t start_hr = high_res_clock();

  int samples[num_samples];
  for (int i = 0; i < num_samples; ++i)
  {
    boost::uint64_t t = high_res_clock();

    asio::error_code ec;
    socket.send_to(asio::buffer(write_buf), target, 0, ec);
    
    do socket.receive(asio::buffer(read_buf), 0, ec);
    while (ec == asio::error::would_block);

    samples[i] = high_res_clock() - t;

    if (target.port() == last_port)
      target.port(first_port);
    else
      target.port(target.port() + 1);
  }

  ptime stop = microsec_clock::universal_time();
  boost::uint64_t stop_hr = high_res_clock();
  boost::uint64_t elapsed_usec = (stop - start).total_microseconds();
  boost::uint64_t elapsed_hr = stop_hr - start_hr;
  double scale = 1.0 * elapsed_usec / elapsed_hr;

  std::sort(samples, samples + num_samples);
  std::printf("  0.0%%\t%f\n", samples[0] * scale);
  std::printf("  0.1%%\t%f\n", samples[num_samples / 1000 - 1] * scale);
  std::printf("  1.0%%\t%f\n", samples[num_samples / 100 - 1] * scale);
  std::printf(" 10.0%%\t%f\n", samples[num_samples / 10 - 1] * scale);
  std::printf(" 20.0%%\t%f\n", samples[num_samples * 2 / 10 - 1] * scale);
  std::printf(" 30.0%%\t%f\n", samples[num_samples * 3 / 10 - 1] * scale);
  std::printf(" 40.0%%\t%f\n", samples[num_samples * 4 / 10 - 1] * scale);
  std::printf(" 50.0%%\t%f\n", samples[num_samples * 5 / 10 - 1] * scale);
  std::printf(" 60.0%%\t%f\n", samples[num_samples * 6 / 10 - 1] * scale);
  std::printf(" 70.0%%\t%f\n", samples[num_samples * 7 / 10 - 1] * scale);
  std::printf(" 80.0%%\t%f\n", samples[num_samples * 8 / 10 - 1] * scale);
  std::printf(" 90.0%%\t%f\n", samples[num_samples * 9 / 10 - 1] * scale);
  std::printf(" 99.0%%\t%f\n", samples[num_samples * 99 / 100 - 1] * scale);
  std::printf(" 99.9%%\t%f\n", samples[num_samples * 999 / 1000 - 1] * scale);
  std::printf("100.0%%\t%f\n", samples[num_samples - 1] * scale);

  double total = 0.0;
  for (int i = 0; i < num_samples; ++i) total += samples[i] * scale;
  std::printf("  mean\t%f\n", total / num_samples);
}
