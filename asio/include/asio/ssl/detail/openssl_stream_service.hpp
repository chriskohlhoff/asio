//
// stream_service.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2005 Voipster / Indrek dot Juhani at voipster dot com
// Copyright (c) 2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_OPENSSL_STREAM_SERVICE_HPP
#define ASIO_SSL_DETAIL_OPENSSL_STREAM_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/ssl/basic_context.hpp"
#include "asio/ssl/stream_base.hpp"
#include "asio/ssl/detail/openssl_operation.hpp"
#include "asio/ssl/detail/openssl_types.hpp"

namespace asio {
namespace ssl {
namespace detail {

template <typename Allocator>
class openssl_stream_service
  : private boost::noncopyable
{
public:
  // The demuxer type.
  typedef basic_demuxer<demuxer_service<Allocator> > demuxer_type;

private:
  //Base handler for asyncrhonous operations
  template <typename Stream>
  class base_handler
  {
  public:
    typedef boost::function<void (const asio::error&, size_t)> func_t;

    base_handler(demuxer_type& d)
      : op_(NULL)
      , demuxer_(d)
      , work_(d)
    {}
    
    void do_func(const asio::error& error, size_t size)
    {
      func_(error, size);
    }
        
    void set_operation(openssl_operation<Stream>* op) { op_ = op; }
    void set_func(func_t func) { func_ = func; }

    ~base_handler()
    {
      delete op_;
    }

  private:
    func_t func_;
    openssl_operation<Stream>* op_;
    demuxer_type& demuxer_;
    typename demuxer_type::work work_;
  };  // class base_handler

  // Handler for asynchronous IO (write/read) operations
  template<typename Stream, typename Handler>
  class io_handler 
    : public base_handler<Stream>
  {
  public:
    io_handler(Handler handler, demuxer_type& d)
      : base_handler<Stream>(d)
      , handler_(handler)
    {
      set_func(boost::bind(
        &io_handler<Stream, Handler>::handler_impl, 
        this, boost::arg<1>(), boost::arg<2>() ));
    }

  private:
    Handler handler_;
    void handler_impl(const asio::error& error, size_t size)
    {
      handler_(error, size);
      delete this;
    }
  };  // class io_handler 

  // Handler for asyncrhonous handshake (connect, accept) functions
  template <typename Stream, typename Handler>
  class handshake_handler
    : public base_handler<Stream>
  {
  public:
    handshake_handler(Handler handler, demuxer_type& d)
      : base_handler<Stream>(d)
      , handler_(handler)
    {
      set_func(boost::bind(
        &handshake_handler<Stream, Handler>::handler_impl, 
        this, boost::arg<1>(), boost::arg<2>() ));
    }

  private:
    Handler handler_;
    void handler_impl(const asio::error& error, size_t)
    {
      handler_(error);
      delete this;
    }

  };  // class handshake_handler

  // Handler for asyncrhonous shutdown
  template <typename Stream, typename Handler>
  class shutdown_handler
    : public base_handler<Stream>
  {
  public:
    shutdown_handler(Handler handler, demuxer_type& d)
      : base_handler<Stream>(d),
        handler_(handler)
    { 
      set_func(boost::bind(
        &shutdown_handler<Stream, Handler>::handler_impl, 
        this, boost::arg<1>(), boost::arg<2>() ));
    }

  private:
    Handler handler_;
    void handler_impl(const asio::error& error, size_t)
    {
      handler_(error);
      delete this;
    }
  };  // class shutdown_handler

public:
  // The implementation type.
  typedef struct impl_struct
  {
    ::SSL* ssl;
    ::BIO* ext_bio;
  } * impl_type;

  // Construct a new stream socket service for the specified demuxer.
  explicit openssl_stream_service(demuxer_type& demuxer)
    : demuxer_(demuxer)
  {
  }

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Return a null stream implementation.
  impl_type null() const
  {
    return 0;
  }

  // Create a new stream implementation.
  template <typename Stream, typename Context_Service>
  void create(impl_type& impl, Stream& next_layer,
      basic_context<Context_Service>& context)
  {
    impl = new impl_struct;
    impl->ssl = ::SSL_new(context.impl());
    ::SSL_set_mode(impl->ssl, SSL_MODE_ENABLE_PARTIAL_WRITE);
    ::BIO* int_bio = 0;
    impl->ext_bio = 0;
    ::BIO_new_bio_pair(&int_bio, 8192, &impl->ext_bio, 8192);
    ::SSL_set_bio(impl->ssl, int_bio, int_bio);
  }

  // Destroy a stream implementation.
  template <typename Stream>
  void destroy(impl_type& impl, Stream& next_layer)
  {
    if (impl != 0)
    {
      ::BIO_free(impl->ext_bio);
      ::SSL_free(impl->ssl);
      delete impl;
      impl = 0;
    }
  }

  // Perform SSL handshaking.
  template <typename Stream, typename Error_Handler>
  void handshake(impl_type& impl, Stream& next_layer,
      stream_base::handshake_type type, Error_Handler error_handler)
  {
    openssl_operation<Stream> op(
      type == stream_base::client ?
        &::SSL_connect:
        &::SSL_accept,
      next_layer,
      impl->ssl,
      impl->ext_bio);
    op.start();
  }

  // Start an asynchronous SSL handshake.
  template <typename Stream, typename Handler>
  void async_handshake(impl_type& impl, Stream& next_layer,
      stream_base::handshake_type type, Handler handler)
  {
    typedef handshake_handler<Stream, Handler> connect_handler;

    connect_handler* local_handler = 
      new connect_handler(handler, demuxer_);

    openssl_operation<Stream>* op = new openssl_operation<Stream>
    (
      type == stream_base::client ?
        &::SSL_connect:
        &::SSL_accept,
      next_layer,
      impl->ssl,
      impl->ext_bio,
      boost::bind
      (
        &base_handler<Stream>::do_func, 
        local_handler,
        boost::arg<1>(),
        boost::arg<2>()
      )
    );
    local_handler->set_operation(op);

    demuxer_.post(boost::bind(&openssl_operation<Stream>::start, op));
  }

  // Shut down SSL on the stream.
  template <typename Stream, typename Error_Handler>
  void shutdown(impl_type& impl, Stream& next_layer,
      Error_Handler error_handler)
  {
    openssl_operation<Stream> op(
      &::SSL_shutdown,
      next_layer,
      impl->ssl,
      impl->ext_bio);
    op.start();
  }

  // Asynchronously shut down SSL on the stream.
  template <typename Stream, typename Handler>
  void async_shutdown(impl_type& impl, Stream& next_layer, Handler handler)
  {
    typedef shutdown_handler<Stream, Handler> disconnect_handler;

    disconnect_handler* local_handler = 
      new disconnect_handler(handler, demuxer_);

    openssl_operation<Stream>* op = new openssl_operation<Stream>
    (
      &::SSL_shutdown,
      next_layer,
      impl->ssl,
      impl->ext_bio,
      boost::bind
      (
        &base_handler<Stream>::do_func, 
        local_handler, 
        boost::arg<1>(),
        boost::arg<2>()
      )
    );
    local_handler->set_operation(op);

    demuxer_.post(boost::bind(&openssl_operation<Stream>::start, op));        
  }

  // Write some data to the stream.
  template <typename Stream, typename Const_Buffers, typename Error_Handler>
  std::size_t write_some(impl_type& impl, Stream& next_layer,
      const Const_Buffers& buffers, Error_Handler error_handler)
  {
    boost::function<int (SSL*)> send_func =
      boost::bind(&::SSL_write, boost::arg<1>(),  
          asio::buffer_cast<const void*>(*buffers.begin()),
          static_cast<int>(asio::buffer_size(*buffers.begin())));
    openssl_operation<Stream> op(
      send_func,
      next_layer,
      impl->ssl,
      impl->ext_bio
    );
    return static_cast<size_t>(op.start());
  }

  // Start an asynchronous write.
  template <typename Stream, typename Const_Buffers, typename Handler>
  void async_write_some(impl_type& impl, Stream& next_layer,
      const Const_Buffers& buffers, Handler handler)
  {
    typedef io_handler<Stream, Handler> send_handler;

    send_handler* local_handler = new send_handler(handler, demuxer_);

    boost::function<int (SSL*)> send_func =
      boost::bind(&::SSL_write, boost::arg<1>(),
          asio::buffer_cast<const void*>(*buffers.begin()),
          static_cast<int>(asio::buffer_size(*buffers.begin())));

    openssl_operation<Stream>* op = new openssl_operation<Stream>
    (
      send_func,
      next_layer,
      impl->ssl,
      impl->ext_bio,
      boost::bind
      (
        &base_handler<Stream>::do_func, 
        local_handler, 
        boost::arg<1>(),
        boost::arg<2>()
      )
    );
    local_handler->set_operation(op);

    demuxer_.post(boost::bind(&openssl_operation<Stream>::start, op));        
  }

  // Read some data from the stream.
  template <typename Stream, typename Mutable_Buffers, typename Error_Handler>
  std::size_t read_some(impl_type& impl, Stream& next_layer,
      const Mutable_Buffers& buffers, Error_Handler error_handler)
  {
    boost::function<int (SSL*)> recv_func =
      boost::bind(&::SSL_read, boost::arg<1>(),
          asio::buffer_cast<void*>(*buffers.begin()),
          asio::buffer_size(*buffers.begin()));
    openssl_operation<Stream> op(recv_func,
      next_layer,
      impl->ssl,
      impl->ext_bio
    );

    return static_cast<size_t>(op.start());
  }

  // Start an asynchronous read.
  template <typename Stream, typename Mutable_Buffers, typename Handler>
  void async_read_some(impl_type& impl, Stream& next_layer,
      const Mutable_Buffers& buffers, Handler handler)
  {
    typedef io_handler<Stream, Handler> recv_handler;

    recv_handler* local_handler = new recv_handler(handler, demuxer_);

    boost::function<int (SSL*)> recv_func =
      boost::bind(&::SSL_read, boost::arg<1>(),
          asio::buffer_cast<void*>(*buffers.begin()),
          asio::buffer_size(*buffers.begin()));

    openssl_operation<Stream>* op = new openssl_operation<Stream>
    (
      recv_func,
      next_layer,
      impl->ssl,
      impl->ext_bio,
      boost::bind
      (
        &base_handler<Stream>::do_func, 
        local_handler, 
        boost::arg<1>(),
        boost::arg<2>()
      )
    );
    local_handler->set_operation(op);

    demuxer_.post(boost::bind(&openssl_operation<Stream>::start, op));        
  }

  // Peek at the incoming data on the stream.
  template <typename Stream, typename Mutable_Buffers, typename Error_Handler>
  std::size_t peek(impl_type& impl, Stream& next_layer,
      const Mutable_Buffers& buffers, Error_Handler error_handler)
  {
    return 0;
  }

  // Determine the amount of data that may be read without blocking.
  template <typename Stream, typename Error_Handler>
  std::size_t in_avail(impl_type& impl, Stream& next_layer,
      Error_Handler error_handler)
  {
    return 0;
  }

private:
  // The demuxer used to dispatch handlers.
  demuxer_type& demuxer_;
};

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_OPENSSL_STREAM_SERVICE_HPP
