//
// Linger_Socket_Option.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Linger_Socket_Option concept.
/**
 * @par Implemented By:
 * asio::socket_base::linger
 */
class Linger_Socket_Option
  : public Socket_Option
{
public:
  /// Default constructor initialises to disabled with a 0 timeout.
  Linger_Socket_Option();

  /// Construct with specific option values.
  Linger_Socket_Option(bool enabled, unsigned short timeout);

  /// Set the value for whether linger is enabled.
  void enabled(bool value);

  /// Get the value for whether linger is enabled.
  bool enabled() const;

  /// Set the value for the linger timeout in seconds.
  void timeout(unsigned short value);

  /// Get the value for the linger timeout in seconds.
  unsigned short timeout() const;
};
