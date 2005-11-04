#include "server.hpp"
#include <boost/bind.hpp>

namespace http {
namespace server {

server::server(short port, const std::string& doc_root)
  : demuxer_(),
    acceptor_(demuxer_),
    connection_manager_(),
    new_connection_(new connection(demuxer_,
          connection_manager_, request_handler_)),
    request_handler_(doc_root)
{
  // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
  asio::ipv4::tcp::endpoint endpoint(port);
  acceptor_.open(endpoint.protocol());
  acceptor_.set_option(asio::socket_acceptor::reuse_address(true));
  acceptor_.bind(endpoint);
  acceptor_.listen();
  acceptor_.async_accept(new_connection_->socket(),
      boost::bind(&server::handle_accept, this,
        asio::placeholders::error));
}

void server::run()
{
  // The demuxer::run() call will block until all asynchronous operations have
  // finished. While the server is running, there is always at least one
  // asynchronous operation outstanding: the asynchronous accept call waiting
  // for new incoming connections.
  demuxer_.run();
}

void server::stop()
{
  // Post a call to the stop function so that server::stop() is safe to call
  // from any thread.
  demuxer_.post(boost::bind(&server::handle_stop, this));
}

void server::handle_accept(const asio::error& e)
{
  if (!e)
  {
    connection_manager_.start(new_connection_);
    new_connection_.reset(new connection(demuxer_,
          connection_manager_, request_handler_));
    acceptor_.async_accept(new_connection_->socket(),
        boost::bind(&server::handle_accept, this,
          asio::placeholders::error));
  }
  else if (e == asio::error::connection_aborted)
  {
    acceptor_.async_accept(new_connection_->socket(),
        boost::bind(&server::handle_accept, this,
          asio::placeholders::error));
  }
}

void server::handle_stop()
{
  // The server is stopped by cancelling all outstanding asynchronous
  // operations. Once all operations have finished the demuxer::run() call will
  // exit.
  acceptor_.close();
  connection_manager_.stop_all();
}

} // namespace server
} // namespace http
