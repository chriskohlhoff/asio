#include <asio.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <vector>
#include "connection.hpp" // Must come before boost/serialization/*.hpp
#include <boost/serialization/vector.hpp>
#include "stock.hpp"

namespace serialization {

/// Serves stock quote information to any client that connects to it.
class server
{
public:
  /// Constructor opens the acceptor and starts waiting for the first incoming
  /// connection.
  server(asio::demuxer& demuxer, unsigned short port)
    : acceptor_(demuxer, asio::ipv4::tcp::endpoint(port))
  {
    // Create the data to be sent to each client.
    stock s;
    s.code = "ABC";
    s.name = "A Big Company";
    s.open_price = 4.56;
    s.high_price = 5.12;
    s.low_price = 4.33;
    s.last_price = 4.98;
    s.buy_price = 4.96;
    s.buy_quantity = 1000;
    s.sell_price = 4.99;
    s.sell_quantity = 2000;
    stocks_.push_back(s);
    s.code = "DEF";
    s.name = "Developer Entertainment Firm";
    s.open_price = 20.24;
    s.high_price = 22.88;
    s.low_price = 19.50;
    s.last_price = 19.76;
    s.buy_price = 19.72;
    s.buy_quantity = 34000;
    s.sell_price = 19.85;
    s.sell_quantity = 45000;
    stocks_.push_back(s);

    // Start an accept operation for a new connection.
    connection_ptr new_conn(new connection(acceptor_.demuxer()));
    acceptor_.async_accept(new_conn->socket(),
        boost::bind(&server::handle_accept, this,
          asio::placeholders::error, new_conn));
  }

  /// Handle completion of a accept operation.
  void handle_accept(const asio::error& e, connection_ptr conn)
  {
    if (!e)
    {
      // Successfully accepted a new connection.
      conn->async_write(stocks_,
          boost::bind(&server::handle_write, this,
            asio::placeholders::error, conn));

      // Start an accept operation for a new connection.
      connection_ptr new_conn(new connection(acceptor_.demuxer()));
      acceptor_.async_accept(new_conn->socket(),
          boost::bind(&server::handle_accept, this,
            asio::placeholders::error, new_conn));
    }
    else if (e == asio::error::connection_aborted)
    {
      // Accept operation failed because a connection was aborted before we were
      // able to accept it. We'll try to accept a new connection using the same
      // socket.
      acceptor_.async_accept(conn->socket(),
          boost::bind(&server::handle_accept, this,
            asio::placeholders::error, conn));
    }
    else
    {
      // Some other error. Log it and return. Since we are not starting a new
      // accept operation the demuxer will run out of work to do and the server
      // will exit.
      std::cerr << e << std::endl;
    }
  }

  /// Handle completion of a write operation.
  void handle_write(const asio::error& e, connection_ptr conn)
  {
    // Nothing to do. The socket will be closed automatically when the last
    // reference to the connection object goes away.
  }

private:
  /// The acceptor object used to accept incoming socket connections.
  asio::socket_acceptor acceptor_;

  /// The data to be sent to each client.
  std::vector<stock> stocks_;
};

} // namespace serialization

int main(int argc, char* argv[])
{
  try
  {
    // Check command line arguments.
    if (argc != 2)
    {
      std::cerr << "Usage: server <port>" << std::endl;
      return 1;
    }
    unsigned short port = boost::lexical_cast<unsigned short>(argv[1]);

    asio::demuxer demuxer;
    serialization::server server(demuxer, port);
    demuxer.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
