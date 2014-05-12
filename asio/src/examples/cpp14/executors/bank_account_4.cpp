#include <asio/dispatch.hpp>
#include <asio/strand_executor.hpp>
#include <asio/system_executor.hpp>
#include <asio/use_future.hpp>
#include <iostream>

using asio::dispatch;
using asio::strand_executor;
using asio::system_executor;
using asio::use_future;

// Active object sharing a system-wide pool of threads.
// The caller chooses how to wait for the operation to finish.
// Lightweight, immediate execution using dispatch.

class bank_account
{
  int balance_ = 0;
  mutable strand_executor<system_executor> ex_;

public:
  template <class CompletionToken>
  auto deposit(int amount, CompletionToken&& token)
  {
    return dispatch(ex_, [=]
      {
        balance_ += amount;
      },
      std::forward<CompletionToken>(token));
  }

  template <class CompletionToken>
  auto withdraw(int amount, CompletionToken&& token)
  {
    return dispatch(ex_, [=]
      {
        if (balance_ >= amount)
          balance_ -= amount;
      },
      std::forward<CompletionToken>(token));
  }

  template <class CompletionToken>
  auto balance(CompletionToken&& token) const
  {
    return dispatch(ex_, [=]
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
