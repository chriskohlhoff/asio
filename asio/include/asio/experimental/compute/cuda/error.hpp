//
// experimental/compute/cuda/error.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_ERROR_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/error_code.hpp"

#include <cuda.h>
#include <system_error>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {

class cuda_error_category : public error_category
{
public:
  cuda_error_category() = default;

  const char* name() const noexcept
  {
    return "cuda";
  }

  std::string message(int value) const
  {
    return cudaGetErrorString( static_cast<cudaError_t>( value ) );
  }
};

inline const cuda_error_category cuda_error_category_instance;

const error_category& cuda_category()
{
  return cuda_error_category_instance;
}

} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)
namespace std {

template <>
struct is_error_code_enum<cudaError_t>
{
  static const bool value = true;
};

} // namespace std
#endif // defined(ASIO_HAS_STD_SYSTEM_ERROR)

asio::error_code make_error_code(cudaError_t e)
{
  return asio::error_code(static_cast<int>(e),
      asio::experimental::compute::cuda::cuda_category());
}

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_ERROR_HPP
