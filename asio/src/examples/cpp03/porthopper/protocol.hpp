//
// protocol.hpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2016 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef PORTHOPPER_PROTOCOL_HPP
#define PORTHOPPER_PROTOCOL_HPP

#include <boost/array.hpp>
#include <asio.hpp>
#include <cstring>
#include <iomanip>
#include <string>
#include <sstream>

// This request is sent by the client to the server over a TCP connection.
// The client uses it to perform three functions:
// - To request that data start being sent to a given port.
// - To request that data is no longer sent to a given port.
// - To change the target port to another.
class control_request
{
public:
  // Construct an empty request. Used when receiving.
  control_request() : data_(control_request_size, '\0')
  {
  }

  // Create a request to start sending data to a given port.
  static const control_request start(unsigned short port)
  {
    return control_request(0, port);
  }

  // Create a request to stop sending data to a given port.
  static const control_request stop(unsigned short port)
  {
    return control_request(port, 0);
  }

  // Create a request to change the port that data is sent to.
  static const control_request change(
      unsigned short old_port, unsigned short new_port)
  {
    return control_request(old_port, new_port);
  }

  // Get the old port. Returns 0 for start requests.
  unsigned short old_port() const
  {
    std::istringstream is(data_);
    unsigned short port = 0;
    is >> std::setw(encoded_port_size) >> std::hex >> port;
    return port;
  }

  // Get the new port. Returns 0 for stop requests.
  unsigned short new_port() const
  {
    std::istringstream is(data_);
    unsigned short port = 0;
	is.ignore(encoded_port_size);
    is >> std::setw(encoded_port_size) >> std::hex >> port;
    return port;
  }

  // Obtain buffers for reading from or writing to a socket.
  boost::array<asio::mutable_buffer, 1> to_buffers()
  {
    boost::array<asio::mutable_buffer, 1> buffers
      = { { asio::buffer(data_) } };
    return buffers;
  }

private:
  // Construct with specified old and new ports.
  control_request(unsigned short old_port_number,
      unsigned short new_port_number)
	  : data_(control_request_size, '\0')
  {
    std::ostringstream os;
    os << std::setw(encoded_port_size) << std::hex << old_port_number;
    os << std::setw(encoded_port_size) << std::hex << new_port_number;
	data_ = os.str();
  }

  // The length in bytes of a control_request and its components.
  enum
  {
    encoded_port_size = 4, // 16-bit port in hex.
    control_request_size = encoded_port_size * 2
  };

  // The encoded request data.
  std::string data_;
};

// This frame is sent from the server to subscribed clients over UDP.
class frame
{
public:
  // The maximum allowable length of the payload.
  enum { payload_size = 32 };

  // Construct an empty frame. Used when receiving.
  frame() : data_(frame_size, '\0')
  {
  }

  // Construct a frame with specified frame number and payload.
  frame(unsigned long frame_number, const std::string& payload_data)
    : data_(frame_size, '\0')
  {
    std::ostringstream os;
    os << std::setw(encoded_number_size) << std::hex << frame_number;
    os << std::setw(payload_size)
      << std::setfill(' ') << payload_data.substr(0, payload_size);
	data_ = os.str();
  }

  // Get the frame number.
  unsigned long number() const
  {
    std::istringstream is(data_);
    unsigned long frame_number = 0;
    is >> std::setw(encoded_number_size) >> std::hex >> frame_number;
    return frame_number;
  }

  // Get the payload data.
  const std::string payload() const
  {
    return std::string(data_.data() + encoded_number_size, payload_size);
  }

  // Obtain buffers for reading from or writing to a socket.
  boost::array<asio::mutable_buffer, 1> to_buffers()
  {
    boost::array<asio::mutable_buffer, 1> buffers
      = { { asio::buffer(data_) } };
    return buffers;
  }

private:
  // The length in bytes of a frame and its components.
  enum
  {
    encoded_number_size = 8, // Frame number in hex.
    frame_size = encoded_number_size + payload_size
  };

  // The encoded frame data.
  std::string data_;
};

#endif // PORTHOPPER_PROTOCOL_HPP
