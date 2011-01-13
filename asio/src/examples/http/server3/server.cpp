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
#include <boost/shared_ptr.hpp>
#include <vector>

namespace http {
namespace server3 {

server::server(const std::string& address, const std::string& port,
    const std::string& doc_root, std::size_t thread_pool_size)
  : thread_pool_size_(thread_pool_size),
    acceptor_(io_service_),
    new_connection_(new connection(io_service_, request_handler_)),
    request_handler_(doc_root)
{
  // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
  asio::ip::tcp::resolver resolver(io_service_);
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
  // Create a pool of threads to run all of the io_services.
  std::vector<boost::shared_ptr<asio::thread> > threads;
  for (std::size_t i = 0; i < thread_pool_size_; ++i)
  {
    boost::shared_ptr<asio::thread> thread(new asio::thread(
          boost::bind(&asio::io_service::run, &io_service_)));
    threads.push_back(thread);
  }

  // Wait for all threads in the pool to exit.
  for (std::size_t i = 0; i < threads.size(); ++i)
    threads[i]->join();
}

void server::stop()
{
  io_service_.stop();
}

void server::handle_accept(const asio::error_code& e)
{
  if (!e)
  {
    new_connection_->start();
    new_connection_.reset(new connection(io_service_, request_handler_));
    acceptor_.async_accept(new_connection_->socket(),
        boost::bind(&server::handle_accept, this,
          asio::placeholders::error));
  }
}

} // namespace server3
} // namespace http
