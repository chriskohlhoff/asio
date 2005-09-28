#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include "asio.hpp"
#include "asio/ssl.hpp"

enum { max_length = 1024 };

class client
{
public:
  client(asio::demuxer& d, const asio::ipv4::tcp::endpoint& server_endpoint)
    : context_(d, asio::ssl::context::sslv23),
      socket_(d, context_)
  {
    socket_.lowest_layer().async_connect(server_endpoint,
        boost::bind(&client::handle_connect, this, asio::placeholders::error));
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

      asio::async_write_n(socket_, asio::buffers(request_, request_length),
          boost::bind(&client::handle_write, this, asio::placeholders::error,
            asio::placeholders::last_bytes_transferred,
            asio::placeholders::total_bytes_transferred));
    }
    else
    {
      std::cout << "Handshake failed: " << error << "\n";
    }
  }

  void handle_write(const asio::error& error, size_t last_bytes_transferred,
      size_t total_bytes_transferred)
  {
    if (!error && last_bytes_transferred > 0)
    {
      asio::async_read_n(socket_,
          asio::buffers(reply_, total_bytes_transferred),
          boost::bind(&client::handle_read, this, asio::placeholders::error,
            asio::placeholders::total_bytes_transferred));
    }
    else
    {
      std::cout << "Write failed: " << error << "\n";
    }
  }

  void handle_read(const asio::error& error, size_t total_bytes_transferred)
  {
    if (!error)
    {
      std::cout << "Reply: ";
      std::cout.write(reply_, total_bytes_transferred);
      std::cout << "\n";
    }
    else
    {
      std::cout << "Read failed: " << error << "\n";
    }
  }

private:
  asio::ssl::context context_;
  asio::ssl::stream<asio::stream_socket> socket_;
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

    asio::demuxer d;

    using namespace std; // For atoi.
    asio::ipv4::host_resolver hr(d);
    asio::ipv4::host h;
    hr.get_host_by_name(h, argv[1]);
    asio::ipv4::tcp::endpoint ep(atoi(argv[2]), h.address(0));

    client c(d, ep);

    d.run();
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
