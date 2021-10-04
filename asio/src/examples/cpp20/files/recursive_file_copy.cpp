#include <asio.hpp>
#include <asio/awaitable.hpp>
#include <asio/experimental/as_tuple.hpp>
#include <asio/experimental/awaitable_operators.hpp>
#include <cstdio>
#include <filesystem>

using namespace asio;
using namespace asio::experimental;
using namespace asio::experimental::awaitable_operators;
namespace fs = std::filesystem;
using stream_file = asio::posix::stream_descriptor;

constexpr std::size_t buf_size = 65536;
constexpr std::size_t buf_align = 512;
constexpr std::size_t active_copies_high_watermark = 500;
constexpr std::size_t active_copies_low_watermark = 400;

stream_file open_file(const any_io_executor& ex, const fs::path& p, int flags, int mode = 0)
{
  int fd = ::open(p.c_str(), flags, mode);
  if (fd < 0)
  {
    int err = errno;
    throw std::system_error(std::error_code(err, std::system_category()));
  }
  return stream_file(ex, fd);
}

stream_file open_file_read_only(const any_io_executor& ex, const fs::path& p)
{
  return open_file(ex, p, O_RDONLY);
}

stream_file open_file_write_only(const any_io_executor& ex, const fs::path& p)
{
  return open_file(ex, p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
}

mutable_buffer align(mutable_buffer buf)
{
  void* data = buf.data();
  std::size_t size = buf.size();
  if (std::align(buf_align, buf_size, data, size) == nullptr)
    std::abort();
  return mutable_buffer(data, size);
}

awaitable<std::size_t> async_copy_file(const fs::path& from, const fs::path& to)
{
  stream_file from_file = open_file_read_only(co_await this_coro::executor, from);
  stream_file to_file = open_file_write_only(co_await this_coro::executor, to);

  std::vector<std::byte> buf_space(buf_size + buf_align);
  auto buf = align(buffer(buf_space));

  std::size_t bytes_copied = 0;
  while (true)
  {
    auto [e, n] = co_await from_file.async_read_some(buf, as_tuple(use_awaitable));
    if (e == stream_errc::eof) break;
    if (e) throw std::system_error(e);
    co_await async_write(to_file, buffer(buf, n), use_awaitable);
    bytes_copied += n;
  }
  co_return bytes_copied;
}

awaitable<std::size_t> copy_one_file(const fs::path& from, const fs::path& to)
{
  try
  {
    auto bytes_copied = co_await async_copy_file(from, to);
    std::printf("copied %ld bytes from %s to %s\n", bytes_copied, from.c_str(), to.c_str());
    co_return bytes_copied;
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "exception copying from %s to %s: \n", from.c_str(), to.c_str());
    co_return 0;
  }
}

awaitable<void> wait_for_turn(steady_timer& turn_timer, std::size_t& active_copies)
{
  while (active_copies >= active_copies_high_watermark)
    co_await turn_timer.async_wait(as_tuple(use_awaitable));
  ++active_copies;
}

void end_turn(steady_timer& turn_timer, std::size_t& active_copies)
{
  if (--active_copies <= active_copies_low_watermark)
    turn_timer.cancel();
}

awaitable<std::size_t> queue_file_copy(const fs::path& from, fs::directory_entry& entry,
    const fs::path& to, steady_timer& turn_timer, std::size_t& active_copies)
{
  if (!entry.is_directory())
  {
    auto relative_source = fs::relative(entry.path(), from);
    auto target_parent_path = to / relative_source.parent_path();
    auto target_parent_file = target_parent_path / entry.path().filename();
    fs::create_directories(target_parent_path);
    co_await wait_for_turn(turn_timer, active_copies);
    auto bytes_copied = co_await copy_one_file(entry.path(), target_parent_file);
    end_turn(turn_timer, active_copies);
    co_return bytes_copied;
  }
  co_return 0;
}

awaitable<std::size_t> copy_files(const fs::path& from,
    std::vector<fs::directory_entry>::iterator first,
    std::vector<fs::directory_entry>::iterator last, const fs::path& to,
    steady_timer& turn_timer, std::size_t& active_copies)
{
  auto n = last - first;
  if (n == 1)
    co_return co_await queue_file_copy(from, *first, to, turn_timer, active_copies);
  else if (n > 1)
  {
    auto [n1, n2] = co_await (
        copy_files(from, first, first + n / 2, to, turn_timer, active_copies)
          && copy_files(from, first + n / 2, last, to, turn_timer, active_copies)
      );
    co_return n1 + n2;
  }
  else
    co_return 0;
}

awaitable<std::size_t> copy_files(const fs::path& from, const fs::path& to)
{
  steady_timer turn_timer(co_await this_coro::executor);
  turn_timer.expires_at(steady_timer::time_point::max());
  std::size_t active_copies = 0;

  std::vector<fs::directory_entry> entries{
      fs::recursive_directory_iterator(from),
      fs::recursive_directory_iterator()};

  co_return co_await copy_files(from, entries.begin(),
      entries.end(), to, turn_timer, active_copies);
}

int main(int argc, const char* argv[])
{
  try
  {
    if (argc != 3)
    {
      std::fprintf(stderr, "Usage: file_copy <from-dir> <to-dir>\n");
      return 1;
    }

    fs::path from = argv[1];
    fs::path to = argv[2];

    asio::io_context ctx(1);

    co_spawn(ctx,
        copy_files(from, to),
        [](std::exception_ptr, std::size_t n)
        {
          std::printf("%ld bytes copied\n", n);
        });

    ctx.run();
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "Exception: %s\n", e.what());
  }
}
