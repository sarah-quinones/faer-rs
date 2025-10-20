#pragma once

#include <cmath>
#include <limits>
#include <ostream>

namespace quad {
struct Quad {
  double x0;
  double x1;

  Quad() = default;
  inline Quad(double x) : x0{x}, x1{} {}
  inline Quad(double x, double y) : x0{x}, x1{y} {}
};

using f128 = Quad;

/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
inline double quick_two_sum(double a, double b, double &err) {
  double s = a + b;
  err = b - (s - a);
  return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
inline double quick_two_diff(double a, double b, double &err) {
  double s = a - b;
  err = (a - s) - b;
  return s;
}

/* Computes fl(a+b) and err(a+b).  */
inline double two_sum(double a, double b, double &err) {
  double s = a + b;
  double bb = s - a;
  err = (a - (s - bb)) + (b - bb);
  return s;
}

/* Computes fl(a-b) and err(a-b).  */
inline double two_diff(double a, double b, double &err) {
  double s = a - b;
  double bb = s - a;
  err = (a - (s - bb)) - (b + bb);
  return s;
}

/* Computes fl(a*b) and err(a*b). */
inline double two_prod(double a, double b, double &err) {
  double p = a * b;
  err = std::fma(a, b, -p);
  return p;
}

inline std::ostream &operator<<(std::ostream &out, const Quad &a) {
  out << a.x0;
  return out;
}

inline Quad operator+(const Quad &a, const Quad &b) {
  double s, e;

  s = two_sum(a.x0, b.x0, e);
  e += (a.x1 + b.x1);
  s = quick_two_sum(s, e, e);
  return Quad{s, e};
}

inline Quad operator-(const Quad &a) { return Quad{-a.x0, -a.x1}; }

inline Quad operator-(const Quad &a, const Quad &b) {
  double s1, s2, t1, t2;
  s1 = two_diff(a.x0, b.x0, s2);
  t1 = two_diff(a.x1, b.x1, t2);
  s2 += t1;
  s1 = quick_two_sum(s1, s2, s2);
  s2 += t2;
  s1 = quick_two_sum(s1, s2, s2);
  return Quad{s1, s2};
}

inline Quad operator*(const Quad &a, const Quad &b) {
  double p1, p2;

  p1 = two_prod(a.x0, b.x0, p2);
  p2 += (a.x0 * b.x1 + a.x1 * b.x0);
  p1 = quick_two_sum(p1, p2, p2);
  return Quad(p1, p2);
}

inline Quad operator/(const Quad &a, const Quad &b) {
  double s1, s2;
  double q1, q2;
  Quad r;

  q1 = a.x0 / b.x0; /* approximate quotient */

  /* compute  this - q1 * dd */
  r = b * q1;
  s1 = two_diff(a.x0, r.x0, s2);
  s2 -= r.x1;
  s2 += a.x1;

  /* get next approximation */
  q2 = (s1 + s2) / b.x0;

  /* renormalize */
  r.x0 = quick_two_sum(q1, q2, r.x1);
  return r;
}

inline Quad &operator+=(Quad &a, const Quad &b) {
  a = a + b;
  return a;
}
inline Quad &operator-=(Quad &a, const Quad &b) {
  a = a - b;
  return a;
}
inline Quad &operator*=(Quad &a, const Quad &b) {
  a = a * b;
  return a;
}
inline Quad &operator/=(Quad &a, const Quad &b) {
  a = a / b;
  return a;
}

inline Quad sqrt(Quad const &a) {
  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto infty = std::numeric_limits<double>::infinity();
  if (a.x0 == 0.0) {
    return Quad{};
  } else if (a.x0 < 0.0) {
    return Quad{nan, nan};
  } else if (a.x0 == infty) {
    return Quad{infty, infty};
  } else {
    auto x = 1.0 / std::sqrt(a.x0);
    auto ax = Quad{a.x0 * x};
    return ax + (a - ax * ax) * Quad{x * 0.5};
  }
}

inline Quad fabs(Quad const &a) { return a.x0 < 0.0 ? -a : a; }
inline Quad abs(Quad const &a) { return a.x0 < 0.0 ? -a : a; }

inline bool isfinite(Quad const &a) { return std::isfinite(a.x0); }
inline bool isinf(Quad const &a) { return std::isinf(a.x0); }
inline bool isnan(Quad const &a) {
  return std::isnan(a.x0) || std::isnan(a.x1);
}

inline bool operator==(const Quad &a, const Quad &b) {
  return a.x0 == b.x0 && a.x1 == b.x1;
}
inline bool operator!=(const Quad &a, const Quad &b) { return !(a == b); }
inline bool operator<(const Quad &a, const Quad &b) {
  return (a.x0 < b.x0) || (a.x0 == b.x0 && a.x1 < b.x1);
}
inline bool operator>(const Quad &a, const Quad &b) {
  return (a.x0 > b.x0) || (a.x0 == b.x0 && a.x1 > b.x1);
}
inline bool operator<=(const Quad &a, const Quad &b) {
  return (a.x0 < b.x0) || (a.x0 == b.x0 && a.x1 <= b.x1);
}
inline bool operator>=(const Quad &a, const Quad &b) {
  return (a.x0 > b.x0) || (a.x0 == b.x0 && a.x1 >= b.x1);
}
} // namespace quad

namespace std {
template <> struct numeric_limits<quad::f128> {
  static constexpr auto is_specialized = true;
  static constexpr auto is_signed = true;
  static constexpr auto is_integer = false;
  static constexpr auto is_exact = false;
  static constexpr auto has_infinity = true;
  static constexpr auto has_quiet_NaN = true;
  static constexpr auto has_signaling_NaN = true;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = std::round_to_nearest;
  static constexpr auto is_iec559 = false;
  static constexpr auto is_bounded = true;
  static constexpr auto is_modulo = false;
  static constexpr auto digits = 100;
  static constexpr auto max_digits10 = 30;

  static quad::f128 epsilon() { return quad::f128{1e-30}; }
  static quad::f128 min() { return quad::f128{1e-200}; }
  static quad::f128 max() { return 1.0 / min(); }
  static quad::f128 quiet_NaN() {
    return quad::f128{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
    };
  }
  static quad::f128 infinity() {
    return quad::f128{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
  }
};
} // namespace std
