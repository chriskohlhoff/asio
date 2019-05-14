Pod::Spec.new do |s|
    s.name     = "asio"
    s.version  = File.read('asio/include/asio/version.hpp').match(/#define ASIO_VERSION \d+ \/\/ (.+)/).captures[0]
    s.summary  = 'Asio C++ Library'
    s.homepage = 'https://think-async.com/Asio/'
    s.license  = 'Boost Software License'
    s.author   = { 'Christopher M. Kohlhoff' => 'chris@kohlhoff.com' }
    s.source   = { :git => 'git@github.com:castlabs/asio.git' }

    s.ios.deployment_target = '8.0'
    s.tvos.deployment_target = '9.0'
    s.requires_arc = false

    s.source_files = ['asio/src/asio.cpp', 'asio/include/**/*.ipp', 'asio/include/**/*.hpp']
    s.public_header_files = ['asio/include/**/*.hpp']
    s.header_mappings_dir = 'asio/include'

    s.pod_target_xcconfig = {
        'USE_HEADERMAP' => 'NO',
        'CLANG_ENABLE_MODULES' => 'NO',
        'GCC_PREPROCESSOR_DEFINITIONS' => 'ASIO_STANDALONE ASIO_SEPARATE_COMPILATION'
    }
end
