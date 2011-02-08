//
// server.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "server.hpp"
#include <boost/bind.hpp>

namespace http {
namespace server2 {

server::server(const std::string& address, const std::string& port,
    const std::string& doc_root, std::size_t io_service_pool_size)
  : io_service_pool_(io_service_pool_size),
    acceptor_(io_service_pool_.get_io_service()),
    new_connection_(new connection(
          io_service_pool_.get_io_service(), request_handler_)),
    request_handler_(doc_root)
{
  // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
  asio::ip::tcp::resolver resolver(acceptor_.io_service());
  asio::ip::tcp::resolver::query query(address, port);
  asio::ip::tcp::endpoint endpoint = *resolver.resolve(query);
  acceptor_.open(endpoint.protocol());
  acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
  acceptor_.bind(endpoint);
  acceptor_.listen();
  acceptor_.async_accept(new_connection_->socket(),
      boost::bind(&server::handle_accept, this,
        asio::placeholders::error));
}

void server::run()
{
  io_service_pool_.run();
}

void server::stop()
{
  io_service_pool_.stop();
}

void server::handle_accept(const asio::error_code& e)
{
  if (!e)
  {
    new_connection_->start();
    new_connection_.reset(new connection(
          io_service_pool_.get_io_service(), request_handler_));
    acceptor_.async_accept(new_connection_->socket(),
        boost::bind(&server::handle_accept, this,
          asio::placeholders::error));
  }
}

} // namespace server2
} // namespace http
