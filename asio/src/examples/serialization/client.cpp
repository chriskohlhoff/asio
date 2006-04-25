#include <asio.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <vector>
#include "connection.hpp" // Must come before boost/serialization headers.
#include <boost/serialization/vector.hpp>
#include "stock.hpp"

namespace s11n_example {

/// Downloads stock quote information from a server.
class client
{
public:
  /// Constructor starts the asynchronous connect operation.
  client(asio::io_service& io_service,
      const std::string& host, const std::string& service)
    : connection_(io_service)
  {
    // Resolve the host name into an IP address.
    asio::ip::tcp::resolver resolver(io_service);
    asio::ip::tcp::resolver::query query(host, service);
    asio::ip::tcp::resolver::iterator endpoint_iterator =
      resolver.resolve(query);

    // Start an asynchronous connect operation.
    connection_.socket().async_connect(*endpoint_iterator,
        boost::bind(&client::handle_connect, this,
          asio::placeholders::error, ++endpoint_iterator));
  }

  /// Handle completion of a connect operation.
  void handle_connect(const asio::error& e,
      asio::ip::tcp::resolver::iterator endpoint_iterator)
  {
    if (!e)
    {
      // Successfully established connection. Start operation to read the list
      // of stocks. The connection::async_read() function will automatically
      // decode the data that is read from the underlying socket.
      connection_.async_read(stocks_,
          boost::bind(&client::handle_read, this,
            asio::placeholders::error));
    }
    else if (endpoint_iterator != asio::ip::tcp::resolver::iterator())
    {
      // Try the next endpoint.
      connection_.socket().close();
      connection_.socket().async_connect(*endpoint_iterator,
          boost::bind(&client::handle_connect, this,
            asio::placeholders::error, ++endpoint_iterator));
    }
    else
    {
      // An error occurred. Log it and return. Since we are not starting a new
      // operation the io_service will run out of work to do and the client will
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

    // Since we are not starting a new operation the io_service will run out of
    // work to do and the client will exit.
  }

private:
  /// The connection to the server.
  connection connection_;

  /// The data received from the server.
  std::vector<stock> stocks_;
};

} // namespace s11n_example

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

    asio::io_service io_service;
    s11n_example::client client(io_service, host, port);
    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
