#include "asio.hpp"
#include "asio/detail/thread.hpp"
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>

using namespace asio;

typedef boost::shared_ptr<stream_socket> stream_socket_ptr;

void tpc_echo_session(stream_socket_ptr sock)
{
  try
  {
    enum { max_length = 8192 };
    char data[max_length];

    sock->set_option(socket_option::recv_buffer_size(max_length));
    sock->set_option(socket_option::send_buffer_size(max_length));

    int length;
    while ((length = recv(*sock, data, max_length)) > 0)
      if (send_n(*sock, data, length) <= 0)
        break;
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception in thread: " << e.what() << "\n";
  }
}

void tpc_echo_server(demuxer& d)
{
  socket_acceptor a(d, inet_address_v4(32123));
  for (;;)
  {
    stream_socket_ptr sock(new stream_socket(d));
    a.accept(*sock);
    detail::thread t(boost::bind(tpc_echo_session, sock));
  }
}

int main()
{
  try
  {
    demuxer d;
    tpc_echo_server(d);
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
