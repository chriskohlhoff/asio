//
// execution.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2017 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_HPP
#define ASIO_EXECUTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/execution/allocator.hpp"
#include "asio/execution/blocking.hpp"
#include "asio/execution/blocking_adaptation.hpp"
#include "asio/execution/bulk_guarantee.hpp"
#include "asio/execution/can_prefer.hpp"
#include "asio/execution/can_query.hpp"
#include "asio/execution/can_require.hpp"
#include "asio/execution/context.hpp"
#include "asio/execution/is_oneway_executor.hpp"
#include "asio/execution/mapping.hpp"
#include "asio/execution/oneway.hpp"
#include "asio/execution/outstanding_work.hpp"
#include "asio/execution/prefer.hpp"
#include "asio/execution/query.hpp"
#include "asio/execution/relationship.hpp"
#include "asio/execution/require.hpp"
#include "asio/execution/single.hpp"

#endif // ASIO_EXECUTION_HPP
