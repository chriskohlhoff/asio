//
// ts/executor.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_TS_EXECUTOR_HPP
#define ASIO_TS_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "../../handler_type.hpp"
#include "../../async_result.hpp"
#include "../../associated_allocator.hpp"
#include "../../execution_context.hpp"
#include "../../is_executor.hpp"
#include "../../associated_executor.hpp"
#include "../../bind_executor.hpp"
#include "../../executor_work_guard.hpp"
#include "../../system_executor.hpp"
#include "../../executor.hpp"
#include "../../dispatch.hpp"
#include "../../post.hpp"
#include "../../defer.hpp"
#include "../../strand.hpp"
#include "../../package.hpp"
#include "../../use_future.hpp"

#endif // ASIO_TS_EXECUTOR_HPP
