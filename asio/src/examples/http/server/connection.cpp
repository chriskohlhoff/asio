#include "connection.hpp"
#include <vector>
#include "boost/bind.hpp"
#include "connection_manager.hpp"
#include "request_handler.hpp"

namespace http {
namespace server {

connection::connection(asio::demuxer& d, connection_manager& manager,
    request_handler& handler)
  : socket_(d),
    connection_manager_(manager),
    request_handler_(handler)
{
}

asio::stream_socket& connection::socket()
{
  return socket_;
}

void connection::start()
{
  asio::async_read(socket_, asio::buffer(buffer_),
      boost::bind(&connection::handle_read, shared_from_this(),
        asio::placeholders::error, asio::placeholders::bytes_transferred));
}

void connection::stop()
{
  socket_.close();
}

void connection::handle_read(const asio::error& e,
    std::size_t bytes_transferred)
{
  if (!e && bytes_transferred > 0)
  {
    boost::tribool result;
    boost::tie(result, boost::tuples::ignore) = request_parser_.parse(
        request_, buffer_.data(), buffer_.data() + bytes_transferred);

    if (result)
    {
      request_handler_.handle_request(request_, reply_);
      asio::async_write_n(socket_, reply_.to_buffers(),
          boost::bind(&connection::handle_write, shared_from_this(),
            asio::placeholders::error));
    }
    else if (!result)
    {
      reply_ = reply::stock_reply(reply::bad_request);
      asio::async_write_n(socket_, reply_.to_buffers(),
          boost::bind(&connection::handle_write, shared_from_this(),
            asio::placeholders::error));
    }
    else
    {
      asio::async_read(socket_, asio::buffer(buffer_),
          boost::bind(&connection::handle_read, shared_from_this(),
            asio::placeholders::error, asio::placeholders::bytes_transferred));
    }
  }
  else if (e != asio::error::operation_aborted)
  {
    connection_manager_.stop(shared_from_this());
  }
}

void connection::handle_write(const asio::error& e)
{
  if (e != asio::error::operation_aborted)
  {
    connection_manager_.stop(shared_from_this());
  }
}

} // namespace server
} // namespace http
