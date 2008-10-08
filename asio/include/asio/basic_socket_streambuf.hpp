//
// basic_socket_streambuf.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_SOCKET_STREAMBUF_HPP
#define ASIO_BASIC_SOCKET_STREAMBUF_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <streambuf>
#include <boost/array.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/utility/base_from_member.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_socket.hpp"
#include "asio/io_service.hpp"
#include "asio/stream_socket_service.hpp"
#include "asio/detail/throw_error.hpp"

#if !defined(ASIO_SOCKET_STREAMBUF_MAX_ARITY)
#define ASIO_SOCKET_STREAMBUF_MAX_ARITY 5
#endif // !defined(ASIO_SOCKET_STREAMBUF_MAX_ARITY)

// A macro that should expand to:
//   template <typename T1, ..., typename Tn>
//   basic_socket_streambuf<Protocol, StreamSocketService>* connect(
//       T1 x1, ..., Tn xn)
//   {
//     init_buffers();
//     asio::error_code ec;
//     this->basic_socket<Protocol, StreamSocketService>::close(ec);
//     typedef typename Protocol::resolver_query resolver_query;
//     resolver_query query(x1, ..., xn);
//     resolve_and_connect(query, ec);
//     return !ec ? this : 0;
//   }
// This macro should only persist within this file.

#define ASIO_PRIVATE_CONNECT_DEF( z, n, data ) \
  template <BOOST_PP_ENUM_PARAMS(n, typename T)> \
  basic_socket_streambuf<Protocol, StreamSocketService>* connect( \
      BOOST_PP_ENUM_BINARY_PARAMS(n, T, x)) \
  { \
    init_buffers(); \
    asio::error_code ec; \
    this->basic_socket<Protocol, StreamSocketService>::close(ec); \
    typedef typename Protocol::resolver_query resolver_query; \
    resolver_query query(BOOST_PP_ENUM_PARAMS(n, x)); \
    resolve_and_connect(query, ec); \
    return !ec ? this : 0; \
  } \
  /**/

namespace asio {

/// Iostream streambuf for a socket.
template <typename Protocol,
    typename StreamSocketService = stream_socket_service<Protocol> >
class basic_socket_streambuf
  : public std::streambuf,
    private boost::base_from_member<io_service>,
    public basic_socket<Protocol, StreamSocketService>
{
public:
  /// The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  /// Construct a basic_socket_streambuf without establishing a connection.
  basic_socket_streambuf()
    : basic_socket<Protocol, StreamSocketService>(
        boost::base_from_member<asio::io_service>::member),
      unbuffered_(false)
  {
    init_buffers();
  }

  /// Destructor flushes buffered data.
  virtual ~basic_socket_streambuf()
  {
    if (pptr() != pbase())
      overflow(traits_type::eof());
  }

  /// Establish a connection.
  /**
   * This function establishes a connection to the specified endpoint.
   *
   * @return \c this if a connection was successfully established, a null
   * pointer otherwise.
   */
  basic_socket_streambuf<Protocol, StreamSocketService>* connect(
      const endpoint_type& endpoint)
  {
    init_buffers();
    asio::error_code ec;
    this->basic_socket<Protocol, StreamSocketService>::close(ec);
    this->basic_socket<Protocol, StreamSocketService>::connect(endpoint, ec);
    return !ec ? this : 0;
  }

#if defined(GENERATING_DOCUMENTATION)
  /// Establish a connection.
  /**
   * This function automatically establishes a connection based on the supplied
   * resolver query parameters. The arguments are used to construct a resolver
   * query object.
   *
   * @return \c this if a connection was successfully established, a null
   * pointer otherwise.
   */
  template <typename T1, ..., typename TN>
  basic_socket_streambuf<Protocol, StreamSocketService>* connect(
      T1 t1, ..., TN tn);
#else
  BOOST_PP_REPEAT_FROM_TO(
      1, BOOST_PP_INC(ASIO_SOCKET_STREAMBUF_MAX_ARITY),
      ASIO_PRIVATE_CONNECT_DEF, _ )
#endif

  /// Close the connection.
  /**
   * @return \c this if a connection was successfully established, a null
   * pointer otherwise.
   */
  basic_socket_streambuf<Protocol, StreamSocketService>* close()
  {
    asio::error_code ec;
    sync();
    this->basic_socket<Protocol, StreamSocketService>::close(ec);
    if (!ec)
      init_buffers();
    return !ec ? this : 0;
  }

protected:
  int_type underflow()
  {
    if (gptr() == egptr())
    {
      asio::error_code ec;
      std::size_t bytes_transferred = this->service.receive(
          this->implementation,
          asio::buffer(asio::buffer(get_buffer_) + putback_max),
          0, ec);
      if (ec)
        return traits_type::eof();
      setg(get_buffer_.begin(), get_buffer_.begin() + putback_max,
          get_buffer_.begin() + putback_max + bytes_transferred);
      return traits_type::to_int_type(*gptr());
    }
    else
    {
      return traits_type::eof();
    }
  }

  int_type overflow(int_type c)
  {
    if (unbuffered_)
    {
      if (traits_type::eq_int_type(c, traits_type::eof()))
      {
        // Nothing to do.
        return traits_type::not_eof(c);
      }
      else
      {
        // Send the single character immediately.
        asio::error_code ec;
        char_type ch = traits_type::to_char_type(c);
        this->service.send(this->implementation,
            asio::buffer(&ch, sizeof(char_type)), 0, ec);
        if (ec)
          return traits_type::eof();
        return c;
      }
    }
    else
    {
      // Send all data in the output buffer.
      asio::const_buffer buffer =
        asio::buffer(pbase(), pptr() - pbase());
      while (asio::buffer_size(buffer) > 0)
      {
        asio::error_code ec;
        std::size_t bytes_transferred = this->service.send(
            this->implementation, asio::buffer(buffer),
            0, ec);
        if (ec)
          return traits_type::eof();
        buffer = buffer + bytes_transferred;
      }
      setp(put_buffer_.begin(), put_buffer_.end());

      // If the new character is eof then our work here is done.
      if (traits_type::eq_int_type(c, traits_type::eof()))
        return traits_type::not_eof(c);

      // Add the new character to the output buffer.
      *pptr() = traits_type::to_char_type(c);
      pbump(1);
      return c;
    }
  }

  int sync()
  {
    return overflow(traits_type::eof());
  }

  std::streambuf* setbuf(char_type* s, std::streamsize n)
  {
    if (pptr() == pbase() && s == 0 && n == 0)
    {
      unbuffered_ = true;
      setp(0, 0);
      return this;
    }

    return 0;
  }

private:
  void init_buffers()
  {
    setg(get_buffer_.begin(),
        get_buffer_.begin() + putback_max,
        get_buffer_.begin() + putback_max);
    if (unbuffered_)
      setp(0, 0);
    else
      setp(put_buffer_.begin(), put_buffer_.end());
  }

  template <typename ResolverQuery>
  void resolve_and_connect(const ResolverQuery& query,
      asio::error_code& ec)
  {
    typedef typename Protocol::resolver resolver_type;
    typedef typename Protocol::resolver_iterator iterator_type;
    resolver_type resolver(
        boost::base_from_member<asio::io_service>::member);
    iterator_type i = resolver.resolve(query, ec);
    if (!ec)
    {
      iterator_type end;
      ec = asio::error::host_not_found;
      while (ec && i != end)
      {
        this->basic_socket<Protocol, StreamSocketService>::close();
        this->basic_socket<Protocol, StreamSocketService>::connect(*i, ec);
        ++i;
      }
    }
  }

  enum { putback_max = 8 };
  enum { buffer_size = 512 };
  boost::array<char, buffer_size> get_buffer_;
  boost::array<char, buffer_size> put_buffer_;
  bool unbuffered_;
};

} // namespace asio

#undef ASIO_PRIVATE_CONNECT_DEF

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_SOCKET_STREAMBUF_HPP
