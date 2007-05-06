//
// connection.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "connection.hpp"
#include <vector>
#include <boost/bind.hpp>
#include "request_handler.hpp"

namespace http {
namespace server3 {

connection::connection(asio::io_service& io_service, request_handler& handler)
  : strand_(io_service),
    socket_(io_service),
    request_handler_(handler)
{
}

asio::ip::tcp::socket& connection::socket()
{
  return socket_;
}

void connection::start()
{
  socket_.async_read_some(asio::buffer(buffer_),
      strand_.wrap(
        boost::bind(&connection::handle_read, shared_from_this(),
          asio::placeholders::error,
          asio::placeholders::bytes_transferred)));
}

void connection::stop()
{
  socket_.close();
}

void connection::handle_read(const asio::error_code& e,
    std::size_t bytes_transferred)
{
  if (!e)
  {
    boost::tribool result;
    boost::tie(result, boost::tuples::ignore) = request_parser_.parse(
        request_, buffer_.data(), buffer_.data() + bytes_transferred);

    if (result)
    {
      request_handler_.handle_request(request_, reply_);
      asio::async_write(socket_, reply_.to_buffers(),
          strand_.wrap(
            boost::bind(&connection::handle_write, shared_from_this(),
              asio::placeholders::error)));
    }
    else if (!result)
    {
      reply_ = reply::stock_reply(reply::bad_request);
      asio::async_write(socket_, reply_.to_buffers(),
          strand_.wrap(
            boost::bind(&connection::handle_write, shared_from_this(),
              asio::placeholders::error)));
    }
    else
    {
      socket_.async_read_some(asio::buffer(buffer_),
          strand_.wrap(
            boost::bind(&connection::handle_read, shared_from_this(),
              asio::placeholders::error,
              asio::placeholders::bytes_transferred)));
    }
  }
  else if (e != asio::error::operation_aborted)
  {
    stop();
  }
}

void connection::handle_write(const asio::error_code& e)
{
  if (e != asio::error::operation_aborted)
  {
    stop();
  }
}

} // namespace server3
} // namespace http
