#ifndef SOCKET_STREAM_HPP
#define SOCKET_STREAM_HPP

#include <boost/iostreams/stream.hpp>
#include <boost/shared_ptr.hpp>
#include "asio.hpp"

namespace io = boost::iostreams;

// Adapt a stream_socket into a device.
class socket_device
  : public io::source
{
public:
  // Constructor.
  socket_device(short port, const char* hostname)
    : demuxer_(new asio::demuxer),
      socket_(new asio::stream_socket(*demuxer_))
  {
    asio::ipv4::host_resolver host_resolver(*demuxer_);
    asio::ipv4::host host;
    host_resolver.get_host_by_name(host, hostname);
    asio::ipv4::tcp::endpoint endpoint(port, host.address(0));
    socket_->connect(endpoint);
  }

  // Read.
  std::streamsize read(char* s, std::streamsize n)
  {
    size_t bytes_read = socket_->read(asio::buffer(s, n));
    if (bytes_read == 0)
      return -1;
    return bytes_read;
  }

  // Write.
  std::streamsize write(const char* s, std::streamsize n)
  {
    return socket_->write(asio::buffer(s, n));
  }

private:
  boost::shared_ptr<asio::demuxer> demuxer_;
  boost::shared_ptr<asio::stream_socket> socket_;
};

// Typedefs for iostreams types.
typedef io::stream_buffer<socket_device> socket_stream_buffer;
typedef io::stream<socket_device> socket_stream;

#endif // SOCKET_STREAM_HPP
