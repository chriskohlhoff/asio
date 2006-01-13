#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include "asio.hpp"
#include "asio/ssl.hpp"

enum { max_length = 1024 };

class client
{
public:
  client(asio::io_service& io_service, asio::ssl::context& context,
      const asio::ipv4::tcp::endpoint& server_endpoint)
    : socket_(io_service, context)
  {
    socket_.lowest_layer().async_connect(server_endpoint,
        boost::bind(&client::handle_connect, this,
          asio::placeholders::error));
  }

  void handle_connect(const asio::error& error)
  {
    if (!error)
    {
      socket_.async_handshake(asio::ssl::stream_base::client,
          boost::bind(&client::handle_handshake, this,
            asio::placeholders::error));
    }
    else
    {
      std::cout << "Connect failed: " << error << "\n";
    }
  }

  void handle_handshake(const asio::error& error)
  {
    if (!error)
    {
      std::cout << "Enter message: ";
      std::cin.getline(request_, max_length);
      size_t request_length = strlen(request_);

      asio::async_write(socket_,
          asio::buffer(request_, request_length),
          boost::bind(&client::handle_write, this,
            asio::placeholders::error,
            asio::placeholders::bytes_transferred));
    }
    else
    {
      std::cout << "Handshake failed: " << error << "\n";
    }
  }

  void handle_write(const asio::error& error, size_t bytes_transferred)
  {
    if (!error)
    {
      asio::async_read(socket_,
          asio::buffer(reply_, bytes_transferred),
          boost::bind(&client::handle_read, this,
            asio::placeholders::error,
            asio::placeholders::bytes_transferred));
    }
    else
    {
      std::cout << "Write failed: " << error << "\n";
    }
  }

  void handle_read(const asio::error& error, size_t bytes_transferred)
  {
    if (!error)
    {
      std::cout << "Reply: ";
      std::cout.write(reply_, bytes_transferred);
      std::cout << "\n";
    }
    else
    {
      std::cout << "Read failed: " << error << "\n";
    }
  }

private:
  asio::ssl::stream<asio::ipv4::tcp::socket> socket_;
  char request_[max_length];
  char reply_[max_length];
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 3)
    {
      std::cerr << "Usage: client <host> <port>\n";
      return 1;
    }

    asio::io_service io_service;

    using namespace std; // For atoi.
    asio::ipv4::host_resolver hr(io_service);
    asio::ipv4::host h;
    hr.get_host_by_name(h, argv[1]);
    asio::ipv4::tcp::endpoint ep(atoi(argv[2]), h.address(0));

    asio::ssl::context ctx(io_service, asio::ssl::context::sslv23);
    ctx.set_verify_mode(asio::ssl::context::verify_peer);
    ctx.load_verify_file("ca.pem");
    client c(io_service, ctx, ep);

    io_service.run();
  }
  catch (asio::error& e)
  {
    std::cerr << e << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
