#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include "asio.hpp"
#include "asio/ssl.hpp"

class session
{
public:
  session(asio::demuxer& d, asio::ssl::context& c)
    : socket_(d, c)
  {
  }

  asio::stream_socket& socket()
  {
    return socket_.lowest_layer();
  }

  void start()
  {
    socket_.async_handshake(asio::ssl::stream_base::server,
        boost::bind(&session::handle_handshake, this,
          asio::placeholders::error));
  }

  void handle_handshake(const asio::error& error)
  {
    if (!error)
    {
      socket_.async_read(asio::buffers(data_, max_length),
          boost::bind(&session::handle_read, this, asio::placeholders::error,
            asio::placeholders::bytes_transferred));
    }
    else
    {
      delete this;
    }
  }

  void handle_read(const asio::error& error, size_t bytes_transferred)
  {
    if (!error && bytes_transferred > 0)
    {
      asio::async_write_n(socket_, asio::buffers(data_, bytes_transferred),
          boost::bind(&session::handle_write, this, asio::placeholders::error,
            asio::placeholders::bytes_transferred));
    }
    else
    {
      delete this;
    }
  }

  void handle_write(const asio::error& error, size_t last_bytes_transferred)
  {
    if (!error && last_bytes_transferred > 0)
    {
      socket_.async_read(asio::buffers(data_, max_length),
          boost::bind(&session::handle_read, this, asio::placeholders::error,
            asio::placeholders::bytes_transferred));
    }
    else
    {
      delete this;
    }
  }

private:
  asio::ssl::stream<asio::stream_socket> socket_;
  enum { max_length = 1024 };
  char data_[max_length];
};

class server
{
public:
  server(asio::demuxer& d, short port)
    : demuxer_(d),
      acceptor_(d, asio::ipv4::tcp::endpoint(port)),
      context_(d, asio::ssl::context::sslv23)
  {
    ::SSL_CTX_use_certificate_chain_file(context_.impl(), "server.pem");
    ::SSL_CTX_use_PrivateKey_file(context_.impl(), "server.pem",
        SSL_FILETYPE_PEM);
    ::SSL_CTX_set_options(context_.impl(),
        SSL_OP_ALL | SSL_OP_NO_SSLv2 | SSL_OP_SINGLE_DH_USE);

    BIO* bio = BIO_new_file("dh512.pem", "r");
    DH* dh512 = PEM_read_bio_DHparams(bio, 0, 0, 0);
    BIO_free(bio);
    ::SSL_CTX_set_tmp_dh(context_.impl(), dh512);

    session* new_session = new session(demuxer_, context_);
    acceptor_.async_accept(new_session->socket(),
        boost::bind(&server::handle_accept, this, new_session,
          asio::placeholders::error));
  }

  void handle_accept(session* new_session, const asio::error& error)
  {
    if (!error)
    {
      new_session->start();
      new_session = new session(demuxer_, context_);
      acceptor_.async_accept(new_session->socket(),
          boost::bind(&server::handle_accept, this, new_session,
            asio::placeholders::error));
    }
    else if (error == asio::error::connection_aborted)
    {
      acceptor_.async_accept(new_session->socket(),
          boost::bind(&server::handle_accept, this, new_session,
            asio::placeholders::error));
    }
    else
    {
      delete new_session;
    }
  }

private:
  asio::demuxer& demuxer_;
  asio::socket_acceptor acceptor_;
  asio::ssl::context context_;
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: server <port>\n";
      return 1;
    }

    asio::demuxer d;

    using namespace std; // For atoi.
    server s(d, atoi(argv[1]));

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
