//
// error_handler.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_ERROR_HANDLER_HPP
#define ASIO_ERROR_HANDLER_HPP

#include "asio/detail/push_options.hpp"

namespace asio {

/// This class is used to indicate a placeholder for the actual error value.
class error_t {};

namespace {

/// This variable is used as a placeholder for the error value.
error_t error;

} // namespace

/// The expression class template is used to allow expressions in an error
/// handler template to be distinguished for the purposes of overloading the
/// || and && operators.
template <typename Expr>
class expression
{
public:
  /// Constructor.
  explicit expression(Expr expr)
    : expr_(expr)
  {
  }

  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    return expr_(err);
  }

private:
  /// The contained expression.
  Expr expr_;
};

/// Create an expression object using template type deduction.
template <typename Expr>
expression<Expr> make_expression(Expr expr)
{
  return expression<Expr>(expr);
}

/// Class template to compare the error for equality with a given value.
template <typename Value>
class value_eq_error
{
public:
  /// Constructor.
  explicit value_eq_error(Value value)
    : value_(value)
  {
  }

  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    return value_ == err;
  }

private:
  /// The value to compare the error against.
  Value value_;
};

/// Compare the error for equality with a given value.
template <typename Value>
expression<value_eq_error<Value> > operator==(Value value, error_t)
{
  return make_expression(value_eq_error<Value>(value));
}

/// Compare the error for equality with a given value.
template <typename Value>
expression<value_eq_error<Value> > operator==(error_t, Value value)
{
  return make_expression(value_eq_error<Value>(value));
}

/// Class template to compare the error for inequality with a given value.
template <typename Value>
class value_neq_error
{
public:
  /// Constructor.
  explicit value_neq_error(Value value)
    : value_(value)
  {
  }

  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    return value_ != err;
  }

private:
  /// The value to compare the error against.
  Value value_;
};

/// Compare the error for inequality with a given value.
template <typename Value>
expression<value_neq_error<Value> > operator!=(Value value, error_t)
{
  return make_expression(value_neq_error<Value>(value));
}

/// Compare the error for inequality with a given value.
template <typename Value>
expression<value_neq_error<Value> > operator!=(error_t, Value value)
{
  return make_expression(value_neq_error<Value>(value));
}

/// Class template to perform logical or on two expressions.
template <typename Expr1, typename Expr2>
class expr_or_expr
{
public:
  /// Constructor.
  explicit expr_or_expr(Expr1 expr1, Expr2 expr2)
    : expr1_(expr1),
      expr2_(expr2)
  {
  }

  // Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    return expr1_(err) || expr2_(err);
  }

private:
  /// The first expression to be included in the logical or.
  Expr1 expr1_;

  /// The second expression to be included in the logical or.
  Expr2 expr2_;
};

/// Perform a logical or on two expressions.
template <typename Expr1, typename Expr2>
expression<expr_or_expr<expression<Expr1>, expression<Expr2> > >
operator||(expression<Expr1> expr1, expression<Expr2> expr2)
{
  return make_expression(
      expr_or_expr<expression<Expr1>, expression<Expr2> >(expr1, expr2));
}

/// Class template to perform logical and on two expressions.
template <typename Expr1, typename Expr2>
class expr_and_expr
{
public:
  /// Constructor.
  explicit expr_and_expr(Expr1 expr1, Expr2 expr2)
    : expr1_(expr1),
      expr2_(expr2)
  {
  }

  // Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    return expr1_(err) && expr2_(err);
  }

private:
  /// The first expression to be included in the logical and.
  Expr1 expr1_;

  /// The second expression to be included in the logical and.
  Expr2 expr2_;
};

/// Perform a logical and on two expressions.
template <typename Expr1, typename Expr2>
expression<expr_and_expr<expression<Expr1>, expression<Expr2> > >
operator&&(expression<Expr1> expr1, expression<Expr2> expr2)
{
  return make_expression(
      expr_and_expr<expression<Expr1>, expression<Expr2> >(expr1, expr2));
}

/// Class to always throw an error.
class throw_error_t
{
public:
  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    throw err;
  }
};

/// Always throw an error.
expression<throw_error_t> throw_error()
{
  return make_expression(throw_error_t());
}

/// Class template to throw an error if an expression is true.
template <typename Expr>
class throw_error_if_t
{
public:
  /// Constructor.
  throw_error_if_t(Expr expr)
    : expr_(expr)
  {
  }

  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    if (expr_(err))
      throw err;
    return false;
  }

private:
  /// The expression which, if true, will result in the error being thrown.
  Expr expr_;
};

/// Throw an error if an expression is true.
template <typename Expr>
expression<throw_error_if_t<Expr> > throw_error_if(Expr expr)
{
  return make_expression(throw_error_if_t<Expr>(expr));
}

/// Class template to always set a variable to the error.
template <typename Target>
class set_error_t
{
public:
  /// Constructor.
  set_error_t(Target& target)
    : target_(target)
  {
  }

  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    target_ = err;
    return true;
  }

private:
  /// The target variable to set to the error.
  Target& target_;
};

/// Set a variable to the error.
template <typename Target>
expression<set_error_t<Target> > set_error(Target& target)
{
  if (target)
    target = Target(); 
  return make_expression(set_error_t<Target>(target));
}

/// Class template to set a variable to the error if an expression is true.
template <typename Target, typename Expr>
class set_error_if_t
{
public:
  /// Constructor.
  set_error_if_t(Target& target, Expr expr)
    : target_(target),
      expr_(expr)
  {
  }

  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    if (expr_(err))
    {
      target_ = err;
      return true;
    }
    return false;
  }

private:
  /// The target variable to set to the error.
  Target& target_;

  /// The expression which, if true, will result in the variable being set.
  Expr expr_;
};

/// Set a variable to the error if an expression is true.
template <typename Target, typename Expr>
expression<set_error_if_t<Target, Expr> >
set_error_if(Target& target, Expr expr)
{
  if (target)
    target = Target(); 
  return make_expression(set_error_if_t<Target, Expr>(target, expr));
}

/// Class template to always log an error to a stream.
template <typename Ostream>
class log_error_t
{
public:
  /// Constructor.
  log_error_t(Ostream& ostream)
    : ostream_(ostream)
  {
  }

  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    ostream_ << err << "\n";
    return true;
  }

private:
  /// The ostream variable to set to the error.
  Ostream& ostream_;
};

/// Set a variable to the error.
template <typename Ostream>
expression<log_error_t<Ostream> > log_error(Ostream& ostream)
{
  return make_expression(log_error_t<Ostream>(ostream));
}

/// Class template to set a variable to the error if an expression is true.
template <typename Ostream, typename Expr>
class log_error_if_t
{
public:
  /// Constructor.
  log_error_if_t(Ostream& ostream, Expr expr)
    : ostream_(ostream),
      expr_(expr)
  {
  }

  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    if (expr_(err))
    {
      ostream_ << err << "\n";
      return true;
    }
    return false;
  }

private:
  /// The ostream variable to set to the error.
  Ostream& ostream_;

  /// The expression which, if true, will result in the variable being set.
  Expr expr_;
};

/// Set a variable to the error if an expression is true.
template <typename Ostream, typename Expr>
expression<log_error_if_t<Ostream, Expr> >
log_error_if(Ostream& ostream, Expr expr)
{
  return make_expression(log_error_if_t<Ostream, Expr>(ostream, expr));
}

/// The default error handler.
class default_error_handler
{
public:
  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    if (err)
      throw err;
    return false;
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ERROR_HANDLER_HPP
