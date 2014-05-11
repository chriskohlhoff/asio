#include <asio/post.hpp>
#include <asio/thread_pool.hpp>
#include <asio/use_future.hpp>
#include <iostream>

using asio::get_executor;
using asio::post;
using asio::thread_pool;
using asio::use_future;

// Traditional active object pattern.
// Member functions block until operation is finished.

class bank_account
{
  int balance_ = 0;
  thread_pool pool_{1};
  mutable thread_pool::executor_type ex_ = get_executor(pool_);

public:
  void deposit(int amount)
  {
    post(ex_, [=]
      {
        balance_ += amount;
      },
      use_future).get();
  }

  void withdraw(int amount)
  {
    post(ex_, [=]
      {
        if (balance_ >= amount)
          balance_ -= amount;
      },
      use_future).get();
  }

  int balance() const
  {
    return post(ex_, [=]
      {
        return balance_;
      },
      use_future).get();
  }
};

int main()
{
  bank_account acct;
  acct.deposit(20);
  acct.withdraw(10);
  std::cout << "balance = " << acct.balance() << "\n";
}
