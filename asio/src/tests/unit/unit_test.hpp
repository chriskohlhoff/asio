//
// unit_test.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef UNIT_TEST_HPP
#define UNIT_TEST_HPP

#include <iostream>

static int unit_test_exit_code = 0;

inline int unit_test(const char* name, void (*func)(void))
{
  std::cout << "INFO: " << name << " started\n";

  try
  {
    func();
  }
  catch (...)
  {
    std::cout << "FAIL: unhandled exception\n";
    unit_test_exit_code = 1;
  }

  std::cout << "INFO: " << name << " ended with exit code ";
  std::cout << unit_test_exit_code << "\n";

  return unit_test_exit_code;
}

#define UNIT_TEST(name) int main() { return unit_test(#name, name); }

inline void unit_test_info(const char* file, int line, const char* msg)
{
  std::cout << "INFO: " << file << "(" << line << "):" << msg << "\n";
}

#define UNIT_TEST_INFO(s) unit_test_info(__FILE__, __LINE__, s)

inline void unit_test_check(bool condition, const char* file, int line,
    const char* msg)
{
  if (!condition)
  {
    std::cout << "FAIL: " << file << "(" << line << "):" << msg << "\n";
    unit_test_exit_code = 1;
  }
}

#define UNIT_TEST_CHECK(c) unit_test_check((c), __FILE__, __LINE__, #c)

#endif // UNIT_TEST_HPP
