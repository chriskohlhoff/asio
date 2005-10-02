#ifndef HTTP_REQUEST_HANDLER_HPP
#define HTTP_REQUEST_HANDLER_HPP

#include <string>
#include "boost/noncopyable.hpp"

namespace http {

class reply;
class request;

class request_handler
  : private boost::noncopyable
{
public:
  // Construct with a directory containing files to be served.
  explicit request_handler(const std::string& doc_root);

  // Handle a request and produce a reply.
  void handle_request(const request& req, reply& rep);

private:
  // The directory containing the files to be served.
  std::string doc_root_;
};

} // namespace http

#endif // HTTP_REQUEST_HANDLER_HPP
