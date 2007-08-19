//
// buffer.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/buffer.hpp"

#include "unit_test.hpp"

//------------------------------------------------------------------------------

// buffer_compile test
// ~~~~~~~~~~~~~~~~~~~
// The following test checks that all overloads of the buffer function compile
// and link correctly. Runtime failures are ignored.

namespace buffer_compile {

using namespace asio;

void test()
{
  try
  {
    char raw_data[1024];
    const char const_raw_data[1024] = "";
    void* void_ptr_data = raw_data;
    const void* const_void_ptr_data = const_raw_data;
    boost::array<char, 1024> array_data;
    const boost::array<char, 1024>& const_array_data_1 = array_data;
    boost::array<const char, 1024> const_array_data_2 = { { 0 } };
    std::vector<char> vector_data(1024);
    const std::vector<char>& const_vector_data = vector_data;
    const std::string string_data(1024, ' ');

    // mutable_buffer constructors.

    mutable_buffer mb1;
    mutable_buffer mb2(void_ptr_data, 1024);
    mutable_buffer mb3(mb1);

    // mutable_buffer functions.

    void* ptr1 = buffer_cast<void*>(mb1);
    (void)ptr1;
    std::size_t size1 = buffer_size(mb1);
    (void)size1;

    // mutable_buffer operators.

    mb1 = mb2 + 128;
    mb1 = 128 + mb2;

    // mutable_buffers_1 constructors.

    mutable_buffers_1 mbc1(mb1);
    mutable_buffers_1 mbc2(mbc1);

    // mutable_buffers_1 functions.

    mutable_buffers_1::const_iterator iter1 = mbc1.begin();
    (void)iter1;
    mutable_buffers_1::const_iterator iter2 = mbc1.end();
    (void)iter2;

    // const_buffer constructors.

    const_buffer cb1;
    const_buffer cb2(const_void_ptr_data, 1024);
    const_buffer cb3(cb1);
    const_buffer cb4(mb1);

    // const_buffer functions.

    const void* ptr2 = buffer_cast<const void*>(cb1);
    (void)ptr2;
    std::size_t size2 = buffer_size(cb1);
    (void)size2;

    // const_buffer operators.

    cb1 = cb2 + 128;
    cb1 = 128 + cb2;

    // const_buffers_1 constructors.

    const_buffers_1 cbc1(cb1);
    const_buffers_1 cbc2(cbc1);

    // const_buffers_1 functions.

    const_buffers_1::const_iterator iter3 = cbc1.begin();
    (void)iter3;
    const_buffers_1::const_iterator iter4 = cbc1.end();
    (void)iter4;

    // buffer function overloads.

    mb1 = buffer(mb2);
    mb1 = buffer(mb2, 128);
    cb1 = buffer(cb2);
    cb1 = buffer(cb2, 128);
    mb1 = buffer(void_ptr_data, 1024);
    cb1 = buffer(const_void_ptr_data, 1024);
    mb1 = buffer(raw_data);
    mb1 = buffer(raw_data, 1024);
    cb1 = buffer(const_raw_data);
    cb1 = buffer(const_raw_data, 1024);
    mb1 = buffer(array_data);
    mb1 = buffer(array_data, 1024);
    cb1 = buffer(const_array_data_1);
    cb1 = buffer(const_array_data_1, 1024);
    cb1 = buffer(const_array_data_2);
    cb1 = buffer(const_array_data_2, 1024);
    mb1 = buffer(vector_data);
    mb1 = buffer(vector_data, 1024);
    cb1 = buffer(const_vector_data);
    cb1 = buffer(const_vector_data, 1024);
    cb1 = buffer(string_data);
    cb1 = buffer(string_data, 1024);
  }
  catch (std::exception&)
  {
  }
}

} // namespace buffer_compile

//------------------------------------------------------------------------------

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("buffer");
  test->add(BOOST_TEST_CASE(&buffer_compile::test));
  return test;
}
