#include "reply.hpp"
#include "boost/lexical_cast.hpp"

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
    return to_buffer(ok);
  case reply::created:
    return to_buffer(created);
  case reply::accepted:
    return to_buffer(accepted);
  case reply::no_content:
    return to_buffer(no_content);
  case reply::multiple_choices:
    return to_buffer(multiple_choices);
  case reply::moved_permanently:
    return to_buffer(moved_permanently);
  case reply::moved_temporarily:
    return to_buffer(moved_temporarily);
  case reply::not_modified:
    return to_buffer(not_modified);
  case reply::bad_request:
    return to_buffer(bad_request);
  case reply::unauthorized:
    return to_buffer(unauthorized);
  case reply::forbidden:
    return to_buffer(forbidden);
  case reply::not_found:
    return to_buffer(not_found);
  case reply::internal_server_error:
    return to_buffer(internal_server_error);
  case reply::not_implemented:
    return to_buffer(not_implemented);
  case reply::bad_gateway:
    return to_buffer(bad_gateway);
  case reply::service_unavailable:
    return to_buffer(service_unavailable);
  default:
    return to_buffer(internal_server_error);
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

namespace stock_replies {

char ok[] = "";
char created[] =
  "<html>"
  "<head><title>Created</title></head>"
  "<body><h1>201 Created</h1></body>"
  "</html>";
char accepted[] =
  "<html>"
  "<head><title>Accepted</title></head>"
  "<body><h1>202 Accepted</h1></body>"
  "</html>";
char no_content[] =
  "<html>"
  "<head><title>No Content</title></head>"
  "<body><h1>204 Content</h1></body>"
  "</html>";
char multiple_choices[] =
  "<html>"
  "<head><title>Multiple Choices</title></head>"
  "<body><h1>300 Multiple Choices</h1></body>"
  "</html>";
char moved_permanently[] =
  "<html>"
  "<head><title>Moved Permanently</title></head>"
  "<body><h1>301 Moved Permanently</h1></body>"
  "</html>";
char moved_temporarily[] =
  "<html>"
  "<head><title>Moved Temporarily</title></head>"
  "<body><h1>302 Moved Temporarily</h1></body>"
  "</html>";
char not_modified[] =
  "<html>"
  "<head><title>Not Modified</title></head>"
  "<body><h1>304 Not Modified</h1></body>"
  "</html>";
char bad_request[] =
  "<html>"
  "<head><title>Bad Request</title></head>"
  "<body><h1>400 Bad Request</h1></body>"
  "</html>";
char unauthorized[] =
  "<html>"
  "<head><title>Unauthorized</title></head>"
  "<body><h1>401 Unauthorized</h1></body>"
  "</html>";
char forbidden[] =
  "<html>"
  "<head><title>Forbidden</title></head>"
  "<body><h1>403 Forbidden</h1></body>"
  "</html>";
char not_found[] =
  "<html>"
  "<head><title>Not Found</title></head>"
  "<body><h1>404 Not Found</h1></body>"
  "</html>";
char internal_server_error[] =
  "<html>"
  "<head><title>Internal Server Error</title></head>"
  "<body><h1>500 Internal Server Error</h1></body>"
  "</html>";
char not_implemented[] =
  "<html>"
  "<head><title>Not Implemented</title></head>"
  "<body><h1>501 Not Implemented</h1></body>"
  "</html>";
char bad_gateway[] =
  "<html>"
  "<head><title>Bad Gateway</title></head>"
  "<body><h1>502 Bad Gateway</h1></body>"
  "</html>";
char service_unavailable[] =
  "<html>"
  "<head><title>Service Unavailable</title></head>"
  "<body><h1>503 Service Unavailable</h1></body>"
  "</html>";

std::string to_string(reply::status_type status)
{
  switch (status)
  {
  case reply::ok:
    return ok;
  case reply::created:
    return created;
  case reply::accepted:
    return accepted;
  case reply::no_content:
    return no_content;
  case reply::multiple_choices:
    return multiple_choices;
  case reply::moved_permanently:
    return moved_permanently;
  case reply::moved_temporarily:
    return moved_temporarily;
  case reply::not_modified:
    return not_modified;
  case reply::bad_request:
    return bad_request;
  case reply::unauthorized:
    return unauthorized;
  case reply::forbidden:
    return forbidden;
  case reply::not_found:
    return not_found;
  case reply::internal_server_error:
    return internal_server_error;
  case reply::not_implemented:
    return not_implemented;
  case reply::bad_gateway:
    return bad_gateway;
  case reply::service_unavailable:
    return service_unavailable;
  default:
    return internal_server_error;
  }
}

} // namespace stock_replies

reply reply::stock_reply(reply::status_type status)
{
  reply rep;
  rep.status = status;
  rep.content = stock_replies::to_string(status);
  rep.headers.resize(2);
  rep.headers[0].name = "Content-Length";
  rep.headers[0].value = boost::lexical_cast<std::string>(rep.content.size());
  rep.headers[1].name = "Content-Type";
  rep.headers[1].value = "text/html";
  return rep;
}

} // namespace http
