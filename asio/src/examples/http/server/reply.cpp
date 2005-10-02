#include "reply.hpp"

namespace http {

namespace status_strings {

char ok[] = "HTTP/1.0 200 OK\r\n";
char created[] = "HTTP/1.0 201 Created\r\n";
char accepted[] = "HTTP/1.0 202 Accepted\r\n";
char no_content[] = "HTTP/1.0 204 No Content\r\n";
char multiple_choices[] = "HTTP/1.0 300 Multiple Choices\r\n";
char moved_permanently[] = "HTTP/1.0 301 Moved Permanently\r\n";
char moved_temporarily[] = "HTTP/1.0 302 Moved Temporarily\r\n";
char not_modified[] = "HTTP/1.0 304 Not Modified\r\n";
char bad_request[] = "HTTP/1.0 400 Bad Request\r\n";
char unauthorized[] = "HTTP/1.0 401 Unauthorized\r\n";
char forbidden[] = "HTTP/1.0 403 Forbidden\r\n";
char not_found[] = "HTTP/1.0 404 Not Found\r\n";
char internal_server_error[] = "HTTP/1.0 500 Internal Server Error\r\n";
char not_implemented[] = "HTTP/1.0 501 Not Implemented\r\n";
char bad_gateway[] = "HTTP/1.0 502 Bad Gateway\r\n";
char service_unavailable[] = "HTTP/1.0 503 Service Unavailable\r\n";

template <std::size_t N>
asio::const_buffer to_buffer(const char (&str)[N])
{
  return asio::buffer(str, N - 1);
}

asio::const_buffer to_buffer(reply::status_type status)
{
  switch (status)
  {
  case reply::ok:
    return to_buffer(status_strings::ok);
  case reply::created:
    return to_buffer(status_strings::created);
  case reply::accepted:
    return to_buffer(status_strings::accepted);
  case reply::no_content:
    return to_buffer(status_strings::no_content);
  case reply::multiple_choices:
    return to_buffer(status_strings::multiple_choices);
  case reply::moved_permanently:
    return to_buffer(status_strings::moved_permanently);
  case reply::moved_temporarily:
    return to_buffer(status_strings::moved_temporarily);
  case reply::not_modified:
    return to_buffer(status_strings::not_modified);
  case reply::bad_request:
    return to_buffer(status_strings::bad_request);
  case reply::unauthorized:
    return to_buffer(status_strings::unauthorized);
  case reply::forbidden:
    return to_buffer(status_strings::forbidden);
  case reply::not_found:
    return to_buffer(status_strings::not_found);
  case reply::internal_server_error:
    return to_buffer(status_strings::internal_server_error);
  case reply::not_implemented:
    return to_buffer(status_strings::not_implemented);
  case reply::bad_gateway:
    return to_buffer(status_strings::bad_gateway);
  case reply::service_unavailable:
    return to_buffer(status_strings::service_unavailable);
  default:
    return to_buffer(status_strings::internal_server_error);
  }
}

} // namespace status_strings

namespace misc_strings {

char name_value_separator[] = { ':', ' ' };
char crlf[] = { '\r', '\n' };

} // namespace misc_strings

std::vector<asio::const_buffer> reply::to_buffers()
{
  std::vector<asio::const_buffer> buffers;
  buffers.push_back(status_strings::to_buffer(status));
  for (std::size_t i = 0; i < headers.size(); ++i)
  {
    header& h = headers[i];
    buffers.push_back(asio::buffer(h.name.data(), h.name.size()));
    buffers.push_back(asio::buffer(misc_strings::name_value_separator));
    buffers.push_back(asio::buffer(h.value.data(), h.value.size()));
    buffers.push_back(asio::buffer(misc_strings::crlf));
  }
  buffers.push_back(asio::buffer(misc_strings::crlf));
  buffers.push_back(asio::buffer(content.data(), content.size()));
  return buffers;
}

reply reply::stock_reply(reply::status_type status)
{
  reply rep;
  rep.status = status;
  rep.headers.resize(1);
  rep.headers.back().name = "Content-Length";
  rep.headers.back().value = "0";
  return rep;
}

} // namespace http
