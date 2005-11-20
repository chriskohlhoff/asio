//
// basic_context.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2005 Voipster / Indrek dot Juhani at voipster dot com
// Copyright (c) 2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_BASIC_CONTEXT_HPP
#define ASIO_SSL_BASIC_CONTEXT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <string>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error_handler.hpp"
#include "asio/service_factory.hpp"
#include "asio/ssl/context_base.hpp"

namespace asio {
namespace ssl {

/// SSL context.
template <typename Service>
class basic_context
  : public context_base,
    private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide context operations.
  typedef Service service_type;

  /// The native implementation type of the locking dispatcher.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this context.
  typedef typename service_type::demuxer_type demuxer_type;

  /// Constructor.
  basic_context(demuxer_type& d, method m)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_, m);
  }

  /// Destructor.
  ~basic_context()
  {
    service_.destroy(impl_);
  }

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * context. This is intended to allow access to context functionality that is
   * not otherwise provided.
   */
  impl_type impl()
  {
    return impl_;
  }

  /// Set options on the context.
  /**
   * This function may be used to configure the SSL options used by the context.
   *
   * @param o A bitmask of options. The available option values are defined in
   * the context_base class. The options are bitwise-ored with any existing
   * value for the options.
   *
   * @throws asio::error Thrown on failure.
   */
  void set_options(options o)
  {
    service_.set_options(impl_, o, throw_error());
  }

  /// Set options on the context.
  /**
   * This function may be used to configure the SSL options used by the context.
   *
   * @param o A bitmask of options. The available option values are defined in
   * the context_base class. The options are bitwise-ored with any existing
   * value for the options.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void set_options(options o, Error_Handler error_handler)
  {
    service_.set_options(impl_, o, error_handler);
  }

  /// Set the peer verification mode.
  /**
   * This function may be used to configure the peer verification mode used by
   * the context.
   *
   * @param v A bitmask of peer verification modes. The available verify_mode
   * values are defined in the context_base class.
   *
   * @throws asio::error Thrown on failure.
   */
  void set_verify_mode(verify_mode v)
  {
    service_.set_verify_mode(impl_, v, throw_error());
  }

  /// Set the peer verification mode.
  /**
   * This function may be used to configure the peer verification mode used by
   * the context.
   *
   * @param v A bitmask of peer verification modes. The available verify_mode
   * values are defined in the context_base class.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void set_verify_mode(verify_mode v, Error_Handler error_handler)
  {
    service_.set_verify_mode(impl_, v, error_handler);
  }

  /// Load a certification authority file for performing verification.
  /**
   * This function is used to load one or more trusted certification authorities
   * from a file.
   *
   * @param filename The name of a file containing certification authority
   * certificates in PEM format.
   *
   * @throws asio::error Thrown on failure.
   */
  void load_verify_file(const std::string& filename)
  {
    service_.load_verify_file(impl_, filename, throw_error());
  }

  /// Load a certification authority file for performing verification.
  /**
   * This function is used to load the certificates for one or more trusted
   * certification authorities from a file.
   *
   * @param filename The name of a file containing certification authority
   * certificates in PEM format.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void load_verify_file(const std::string& filename,
      Error_Handler error_handler)
  {
    service_.load_verify_file(impl_, filename, error_handler);
  }

  /// Add a directory containing certificate authority files to be used for
  /// performing verification.
  /**
   * This function is used to specify the name of a directory containing
   * certification authority certificates. Each file in the directory must
   * contain a single certificate. The files must be named using the subject
   * name's hash and an extension of ".0".
   *
   * @param path The name of a directory containing the certificates.
   *
   * @throws asio::error Thrown on failure.
   */
  void add_verify_path(const std::string& path)
  {
    service_.add_verify_path(impl_, path, throw_error());
  }

  /// Add a directory containing certificate authority files to be used for
  /// performing verification.
  /**
   * This function is used to specify the name of a directory containing
   * certification authority certificates. Each file in the directory must
   * contain a single certificate. The files must be named using the subject
   * name's hash and an extension of ".0".
   *
   * @param path The name of a directory containing the certificates.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void add_verify_path(const std::string& path, Error_Handler error_handler)
  {
    service_.add_verify_path(impl_, path, error_handler);
  }

  /// Use a certificate from a file.
  /**
   * This function is used to load a certificate into the context from a file.
   *
   * @param filename The name of the file containing the certificate.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @throws asio::error Thrown on failure.
   */
  void use_certificate_file(const std::string& filename, file_format format)
  {
    service_.use_certificate_file(impl_, filename, format, throw_error());
  }

  /// Use a certificate from a file.
  /**
   * This function is used to load a certificate into the context from a file.
   *
   * @param filename The name of the file containing the certificate.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void use_certificate_file(const std::string& filename, file_format format,
      Error_Handler error_handler)
  {
    service_.use_certificate_file(impl_, filename, format, error_handler);
  }

  /// Use a certificate chain from a file.
  /**
   * This function is used to load a certificate chain into the context from a
   * file.
   *
   * @param filename The name of the file containing the certificate. The file
   * must use the PEM format.
   *
   * @throws asio::error Thrown on failure.
   */
  void use_certificate_chain_file(const std::string& filename)
  {
    service_.use_certificate_chain_file(impl_, filename, throw_error());
  }

  /// Use a certificate chain from a file.
  /**
   * This function is used to load a certificate chain into the context from a
   * file.
   *
   * @param filename The name of the file containing the certificate. The file
   * must use the PEM format.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void use_certificate_chain_file(const std::string& filename,
      Error_Handler error_handler)
  {
    service_.use_certificate_chain_file(impl_, filename, error_handler);
  }

  /// Use a private key from a file.
  /**
   * This function is used to load a private key into the context from a file.
   *
   * @param filename The name of the file containing the private key.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @throws asio::error Thrown on failure.
   */
  void use_private_key_file(const std::string& filename, file_format format)
  {
    service_.use_private_key_file(impl_, filename, format, throw_error());
  }

  /// Use a private key from a file.
  /**
   * This function is used to load a private key into the context from a file.
   *
   * @param filename The name of the file containing the private key.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void use_private_key_file(const std::string& filename, file_format format,
      Error_Handler error_handler)
  {
    service_.use_private_key_file(impl_, filename, format, error_handler);
  }

  /// Use an RSA private key from a file.
  /**
   * This function is used to load an RSA private key into the context from a
   * file.
   *
   * @param filename The name of the file containing the RSA private key.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @throws asio::error Thrown on failure.
   */
  void use_rsa_private_key_file(const std::string& filename, file_format format)
  {
    service_.use_rsa_private_key_file(impl_, filename, format, throw_error());
  }

  /// Use an RSA private key from a file.
  /**
   * This function is used to load an RSA private key into the context from a
   * file.
   *
   * @param filename The name of the file containing the RSA private key.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void use_rsa_private_key_file(const std::string& filename, file_format format,
      Error_Handler error_handler)
  {
    service_.use_rsa_private_key_file(impl_, filename, format, error_handler);
  }

  /// Use the specified file to obtain the temporary Diffie-Hellman parameters.
  /**
   * This function is used to load Diffie-Hellman parameters into the context
   * from a file.
   *
   * @param filename The name of the file containing the Diffie-Hellman
   * parameters. The file must use the PEM format.
   *
   * @throws asio::error Thrown on failure.
   */
  void use_tmp_dh_file(const std::string& filename)
  {
    service_.use_tmp_dh_file(impl_, filename, throw_error());
  }

  /// Use the specified file to obtain the temporary Diffie-Hellman parameters.
  /**
   * This function is used to load Diffie-Hellman parameters into the context
   * from a file.
   *
   * @param filename The name of the file containing the Diffie-Hellman
   * parameters. The file must use the PEM format.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void use_tmp_dh_file(const std::string& filename, Error_Handler error_handler)
  {
    service_.use_tmp_dh_file(impl_, filename, error_handler);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_BASIC_CONTEXT_HPP
