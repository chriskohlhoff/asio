#include <cassert>
#include <iostream>
#include <sstream>
#include "asio.hpp"

using namespace asio;

int main(int argc, char* argv[])
{
  try
  {
    demuxer d;

    stream_socket s(d);
    socket_connector c(d);

    socket_error expected_err;
    c.connect(s, inet_address_v4(321, "0.0.0.0"), set_error(expected_err));

    std::ostringstream os;
    c.connect(s, inet_address_v4(321, "0.0.0.0"), log_error(os));
    assert(!os.str().empty());

    os.str("");
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error == expected_err || log_error(os));
    assert(os.str().empty());

    os.str("");
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error == expected_err && log_error(os));
    assert(!os.str().empty());

    os.str("");
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error != expected_err || log_error(os));
    assert(!os.str().empty());

    os.str("");
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error != expected_err && log_error(os));
    assert(os.str().empty());

    os.str("");
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        log_error_if(os, error == expected_err));
    assert(!os.str().empty());

    os.str("");
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        log_error_if(os, error != expected_err));
    assert(os.str().empty());

    try
    {
      c.connect(s, inet_address_v4(321, "0.0.0.0"), throw_error());
      assert(0);
    }
    catch (socket_error&)
    {
    }

    try
    {
      c.connect(s, inet_address_v4(321, "0.0.0.0"),
          error == expected_err || throw_error());
    }
    catch (socket_error&)
    {
      assert(0);
    }

    try
    {
      c.connect(s, inet_address_v4(321, "0.0.0.0"),
          error == expected_err && throw_error());
      assert(0);
    }
    catch (socket_error&)
    {
    }

    try
    {
      c.connect(s, inet_address_v4(321, "0.0.0.0"),
          error != expected_err || throw_error());
      assert(0);
    }
    catch (socket_error&)
    {
    }

    try
    {
      c.connect(s, inet_address_v4(321, "0.0.0.0"),
          error != expected_err && throw_error());
    }
    catch (socket_error&)
    {
      assert(0);
    }

    try
    {
      c.connect(s, inet_address_v4(321, "0.0.0.0"),
          throw_error_if(error == expected_err));
      assert(0);
    }
    catch (socket_error&)
    {
    }

    try
    {
      c.connect(s, inet_address_v4(321, "0.0.0.0"),
          throw_error_if(error != expected_err));
    }
    catch (socket_error&)
    {
      assert(0);
    }

    socket_error err;
    c.connect(s, inet_address_v4(321, "0.0.0.0"), set_error(err));
    assert(err == expected_err);

    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error == expected_err || set_error(err));
    assert(err != expected_err);

    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error == expected_err && set_error(err));
    assert(err == expected_err);

    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error != expected_err || set_error(err));
    assert(err == expected_err);

    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error != expected_err && set_error(err));
    assert(err != expected_err);

    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        set_error_if(err, error == expected_err));
    assert(err == expected_err);

    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        set_error_if(err, error != expected_err));
    assert(err != expected_err);
  }
  catch (socket_error& e)
  {
    std::cerr << "Unhandled socket error: " << e.message() << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Unhandled exception: " << e.what() << "\n";
  }

  return 0;
}
