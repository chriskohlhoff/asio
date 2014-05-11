#include <asio/post.hpp>
#include <asio/thread_pool.hpp>
#include <asio/use_future.hpp>
#include <iostream>

using asio::async_result;
using asio::get_executor;
using asio::handler_type;
using asio::post;
using asio::thread_pool;
using asio::use_future;

// Traditional active object pattern.
// The caller chooses how to wait for the operation to finish.

class bank_account
{
  int balance_ = 0;
  thread_pool pool_{1};
  mutable thread_pool::executor_type ex_ = get_executor(pool_);

public:
  template <class CompletionToken>
  auto deposit(int amount, CompletionToken&& token)
    -> typename async_result<
        typename handler_type<
          CompletionToken, void()>::type>::type
  {
    return post(ex_, [=]
      {
        balance_ += amount;
      },
      std::forward<CompletionToken>(token));
  }

  template <class CompletionToken>
  auto withdraw(int amount, CompletionToken&& token)
    -> typename async_result<
        typename handler_type<
          CompletionToken, void()>::type>::type
  {
    return post(ex_, [=]
      {
        if (balance_ >= amount)
          balance_ -= amount;
      },
      std::forward<CompletionToken>(token));
  }

  template <class CompletionToken>
  auto balance(CompletionToken&& token) const
    -> typename async_result<
        typename handler_type<
          CompletionToken, void(int)>::type>::type
  {
    return post(ex_, [=]
      {
        return balance_;
      },
      std::forward<CompletionToken>(token));
  }
};

int main()
{
  bank_account acct1;
  acct1.deposit(20, []{ std::cout << "deposit complete\n"; });
  acct1.withdraw(10, []{ std::cout << "withdraw complete\n"; });
  acct1.balance([](int b){ std::cout << "balance = " << b << "\n"; });

  bank_account acct2;
  acct2.deposit(40, use_future).get();
  acct2.withdraw(15, use_future).get();
  std::cout << "balance = " << acct2.balance(use_future).get() << "\n";
}
