
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace std;

using Vec = array<double, 3>;

constexpr double PI = 3.14159265358979323846;
constexpr double ROOT_EPS = 1e-11;

// 多項式の評価（係数は低次から高次の順）
double eval(const vector<double>& p, double x) {
    double v = 0.0;
    for (int i = int(p.size()) - 1; i >= 0; --i)
        v = v * x + p[i];
    return v;
}

// 重複する根を取り除く
void unique_sort(vector<double>& v) {
    sort(v.begin(), v.end());
    vector<double> result;
    for (double x : v) {
        if (result.empty() ||
            abs(x - result.back()) > ROOT_EPS)
            result.push_back(x);
    }
    v.swap(result);
}

// 区間 [lo, hi] にある実根を求める
// 導関数によって単調区間に分割する
vector<double> roots(vector<double> p,
                     double lo, double hi) {
    while (p.size() > 1 && p.back() == 0.0)
        p.pop_back();

    int n = int(p.size()) - 1;
    if (n <= 0) return {};

    if (n == 1) {
        double x = -p[0] / p[1];
        if (lo <= x && x <= hi) return {x};
        return {};
    }

    vector<double> dp(n);
    for (int i = 1; i <= n; ++i)
        dp[i - 1] = i * p[i];

    vector<double> cuts = {lo};
    auto critical = roots(dp, lo, hi);
    cuts.insert(cuts.end(),
                critical.begin(), critical.end());
    cuts.push_back(hi);
    unique_sort(cuts);

    vector<double> result;

    double scale = 1.0;
    for (double c : p)
        scale += abs(c);

    double tol = 1e-12 * scale;

    for (double x : cuts) {
        if (abs(eval(p, x)) <= tol)
            result.push_back(x);
    }

    for (size_t i = 1; i < cuts.size(); ++i) {
        double l = cuts[i - 1];
        double r = cuts[i];

        double fl = eval(p, l);
        double fr = eval(p, r);

        if ((fl < 0 && fr < 0) ||
            (fl > 0 && fr > 0) ||
            fl == 0 || fr == 0)
            continue;

        for (int j = 0; j < 90; ++j) {
            double m = (l + r) / 2.0;
            double fm = eval(p, m);

            if ((fl < 0 && fm < 0) ||
                (fl > 0 && fm > 0)) {
                l = m;
                fl = fm;
            } else {
                r = m;
            }
        }
        result.push_back((l + r) / 2.0);
    }

    unique_sort(result);
    return result;
}

// 弧長の原始関数
// y = a(x-h)^2+k, y'=2a(x-h)
double primitive(double x, double a, double h) {
    double u = 2.0 * a * (x - h);
    return (u * hypot(1.0, u) + asinh(u))
           / (4.0 * a);
}

// 円盤内の弧長を計算
double arc_length(double a, double h, double k) {
    if (!(a > 0.0) ||
        !isfinite(a) || !isfinite(h) || !isfinite(k))
        return 0.0;

    // y = ax^2 + bx + c
    double b = -2.0 * a * h;
    double c = a * h * h + k;

    // x^2 + (ax^2+bx+c)^2 - 1 = 0
    vector<double> p = {
        c*c - 1.0,
        2.0*b*c,
        b*b + 2.0*a*c + 1.0,
        2.0*a*b,
        a*a
    };

    auto rr = roots(p, -1.0, 1.0);

    vector<double> cuts = {-1.0};
    cuts.insert(cuts.end(), rr.begin(), rr.end());
    cuts.push_back(1.0);
    unique_sort(cuts);

    double L = 0.0;

    for (size_t i = 1; i < cuts.size(); ++i) {
        double l = cuts[i - 1];
        double r = cuts[i];
        double mid = (l + r) / 2.0;

        double y = a * (mid - h) * (mid - h) + k;

        if (mid*mid + y*y <= 1.0) {
            L += primitive(r, a, h)
               - primitive(l, a, h);
        }
    }

    return L;
}

// 最適化変数:
// z[0] = log(a), z[1] = h, z[2] = k
// 以下は数値探索上の制限であり、数学的な制約ではない
double objective(const Vec& z) {
    if (z[0] < -12.0 || z[0] > 12.0 ||
        abs(z[1]) > 2.0 ||
        abs(z[2]) > 3.0)
        return -1e6;

    double a = exp(z[0]);
    return arc_length(a, z[1], z[2]);
}

struct Result {
    Vec z;
    double value;
};

// Nelder-Mead法による最大化
Result nelder_mead(Vec start,
                   int max_iter = 2500) {
    constexpr int N = 3;

    array<Vec, N + 1> x;
    array<double, N + 1> f;

    x[0] = start;

    Vec step = {0.25, 0.12, 0.12};

    for (int i = 0; i < N; ++i) {
        x[i + 1] = start;
        x[i + 1][i] += step[i];
    }

    for (int i = 0; i <= N; ++i)
        f[i] = objective(x[i]);

    for (int iter = 0; iter < max_iter; ++iter) {
        array<int, N + 1> order = {0, 1, 2, 3};

        sort(order.begin(), order.end(),
             [&](int i, int j) {
                 return f[i] > f[j];
             });

        auto oldx = x;
        auto oldf = f;

        for (int i = 0; i <= N; ++i) {
            x[i] = oldx[order[i]];
            f[i] = oldf[order[i]];
        }

        double diameter = 0.0;
        for (int i = 1; i <= N; ++i)
            for (int j = 0; j < N; ++j)
                diameter = max(
                    diameter, abs(x[i][j] - x[0][j]));

        if (diameter < 1e-9 &&
            abs(f[0] - f[N]) < 1e-10)
            break;

        Vec center = {0, 0, 0};
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                center[j] += x[i][j] / N;

        auto move = [&](double factor) {
            Vec v;
            for (int j = 0; j < N; ++j)
                v[j] = center[j]
                     + factor * (x[N][j] - center[j]);
            return v;
        };

        // Reflection
        Vec xr = move(-1.0);
        double fr = objective(xr);

        if (fr > f[0]) {
            Vec xe = move(-2.0);
            double fe = objective(xe);
            x[N] = fe > fr ? xe : xr;
            f[N] = max(fe, fr);
        }
        else if (fr > f[N - 1]) {
            x[N] = xr;
            f[N] = fr;
        }
        else {
            // Contraction
            Vec xc = (fr > f[N])
                   ? move(-0.5)
                   : move(0.5);

            double fc = objective(xc);

            if (fc > max(fr, f[N]) ||
                (fr <= f[N] && fc > f[N])) {
                x[N] = xc;
                f[N] = fc;
            }
            else {
                // Shrink
                for (int i = 1; i <= N; ++i) {
                    for (int j = 0; j < N; ++j)
                        x[i][j] =
                            (x[i][j] + x[0][j]) / 2.0;
                    f[i] = objective(x[i]);
                }
            }
        }
    }

    int best = 0;
    for (int i = 1; i <= N; ++i)
        if (f[i] > f[best]) best = i;

    return {x[best], f[best]};
}

int main() {
    mt19937_64 rng(123456789);

    uniform_real_distribution<double> da(-5.0, 8.0);
    uniform_real_distribution<double> dh(-1.0, 1.0);
    uniform_real_distribution<double> dk(-1.5, 1.5);

    Result best{{0, 0, 0}, -1e100};

    // 既知の放物線を初期値に含める
    vector<Vec> initial = {
        {log(100.0), 0.0, -1.0},
        {log(10.0),  0.0, -1.0},
        {log(1.0),   0.0, -1.0},
        {log(0.5),   0.0,  0.0}
    };

    // ランダムな初期値を追加
    for (int i = 0; i < 300; ++i)
        initial.push_back({da(rng), dh(rng), dk(rng)});

    cout << setprecision(15);

    for (size_t i = 0; i < initial.size(); ++i) {
        Result r = nelder_mead(initial[i]);

        if (r.value > best.value) {
            best = r;

            cout << "restart = " << i
                 << "  L = " << best.value
                 << "  a = " << exp(best.z[0])
                 << "  h = " << best.z[1]
                 << "  k = " << best.z[2]
                 << '\n';
        }
    }

    cout << "\nBest result\n";
    cout << "L = " << best.value << '\n';
    cout << "a = " << exp(best.z[0]) << '\n';
    cout << "h = " << best.z[1] << '\n';
    cout << "k = " << best.z[2] << '\n';

    cout << "Upper bound 2*pi = " << 2.0*PI << '\n';
}
