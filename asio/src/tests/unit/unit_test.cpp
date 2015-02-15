//
// unit_test.cpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "unit_test.hpp"

#if !defined(ASIO_STANDALONE)
# if (BOOST_VERSION < 104800)
#  include <boost/test/included/unit_test_framework.hpp>
# else
#  include <boost/test/included/unit_test.hpp>
# endif
#endif
