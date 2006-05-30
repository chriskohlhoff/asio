#ifndef SOCKET_STREAM_HPP
#define SOCKET_STREAM_HPP

#include <boost/iostreams/stream.hpp>
#include <boost/shared_ptr.hpp>
#include <string>
#include "asio.hpp"

namespace io = boost::iostreams;

// Adapt a stream_socket into a device.
class socket_device
{
public:
  // Typedefs for Boost.Iostreams.
  typedef char char_type;
  typedef io::bidirectional_device_tag category;

  // Construct without connecting.
  socket_device(int /*ignored*/)
    : io_service_(new asio::io_service),
      socket_(new asio::ip::tcp::socket(*io_service_))
  {
  }

  // Constructor by opening a connection.
  socket_device(const std::string& host, const std::string& service)
    : io_service_(new asio::io_service),
      socket_(new asio::ip::tcp::socket(*io_service_))
  {
    // Get a list of endpoints that match the supplied host and service.
    asio::ip::tcp::resolver resolver(*io_service_);
    asio::ip::tcp::resolver::query query(host, service);
    asio::ip::tcp::resolver::iterator iterator = resolver.resolve(query);

    // Try each endpoint in the list until we get one that works.
    asio::error error(asio::error::host_not_found);
    while (error && iterator != asio::ip::tcp::resolver::iterator())
    {
      error = 0;
      socket_->close();
      socket_->connect(*iterator, asio::assign_error(error));
      ++iterator;
    }
    if (error)
      throw error;
  }

  // Get underlying socket.
  asio::ip::tcp::socket& socket()
  {
    return *socket_;
  }

  // Read.
  std::streamsize read(char* s, std::streamsize n)
  {
    asio::error error;
    size_t bytes_read = socket_->read_some(
        asio::buffer(s, n), asio::assign_error(error));
    if (error == asio::error::eof)
      return -1;
    else if (error)
      throw error;
    return bytes_read;
  }

  // Write.
  std::streamsize write(const char* s, std::streamsize n)
  {
    return socket_->write_some(asio::buffer(s, n));
  }

private:
  boost::shared_ptr<asio::io_service> io_service_;
  boost::shared_ptr<asio::ip::tcp::socket> socket_;
};

// Typedefs for iostreams types.
typedef io::stream_buffer<socket_device> socket_stream_buffer;
typedef io::stream<socket_device> socket_stream;

#endif // SOCKET_STREAM_HPP
