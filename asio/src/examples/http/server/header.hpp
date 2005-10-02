#ifndef HTTP_HEADER_HPP
#define HTTP_HEADER_HPP

#include <string>

namespace http {

struct header
{
  std::string name;
  std::string value;
};

} // namespace http

#endif // HTTP_HEADER_HPP
