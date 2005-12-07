#include <asio.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <vector>
#include "connection.hpp" // Must come before boost/serialization/*.hpp
#include <boost/serialization/vector.hpp>
#include "stock.hpp"

namespace serialization {

/// Downloads stock quote information from a server.
class client
{
public:
  /// Constructor starts the asynchronous connect operation.
  client(asio::demuxer& demuxer,
      const std::string& hostname, unsigned short port)
    : connection_(demuxer)
  {
    // Resolve the host name into an IP address.
    asio::ipv4::host_resolver host_resolver(demuxer);
    asio::ipv4::host host;
    host_resolver.get_host_by_name(host, hostname);

    // Start an asynchronous connect operation.
    asio::ipv4::tcp::endpoint endpoint(port, host.address(0));
    connection_.socket().async_connect(endpoint,
        boost::bind(&client::handle_connect, this,
          asio::placeholders::error));
  }

  /// Handle completion of a connect operation.
  void handle_connect(const asio::error& e)
  {
    if (!e)
    {
      // Successfully established connection. Start read operation.
      connection_.async_read(stocks_,
          boost::bind(&client::handle_read, this,
            asio::placeholders::error));
    }
    else
    {
      // An error occurred. Log it and return. Since we are not starting a new
      // operation the demuxer will run out of work to do and the client will
      // exit.
      std::cerr << e << std::endl;
    }
  }

  /// Handle completion of a read operation.
  void handle_read(const asio::error& e)
  {
    if (!e)
    {
      // Print out the data that was received.
      for (std::size_t i = 0; i < stocks_.size(); ++i)
      {
        std::cout << "Stock number " << i << "\n";
        std::cout << "  code: " << stocks_[i].code << "\n";
        std::cout << "  name: " << stocks_[i].name << "\n";
        std::cout << "  open_price: " << stocks_[i].open_price << "\n";
        std::cout << "  high_price: " << stocks_[i].high_price << "\n";
        std::cout << "  low_price: " << stocks_[i].low_price << "\n";
        std::cout << "  last_price: " << stocks_[i].last_price << "\n";
        std::cout << "  buy_price: " << stocks_[i].buy_price << "\n";
        std::cout << "  buy_quantity: " << stocks_[i].buy_quantity << "\n";
        std::cout << "  sell_price: " << stocks_[i].sell_price << "\n";
        std::cout << "  sell_quantity: " << stocks_[i].sell_quantity << "\n";
      }
    }
    else
    {
      // An error occurred.
      std::cerr << e << std::endl;
    }

    // Since we are not starting a new operation the demuxer will run out of
    // work to do and the client will exit.
  }

private:
  /// The connection to the server.
  connection connection_;

  /// The data received from the server.
  std::vector<stock> stocks_;
};

} // namespace serialization

int main(int argc, char* argv[])
{
  try
  {
    // Check command line arguments.
    if (argc != 3)
    {
      std::cerr << "Usage: client <host> <port>" << std::endl;
      return 1;
    }
    std::string host = argv[1];
    unsigned short port = boost::lexical_cast<unsigned short>(argv[2]);

    asio::demuxer demuxer;
    serialization::client client(demuxer, host, port);
    demuxer.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
