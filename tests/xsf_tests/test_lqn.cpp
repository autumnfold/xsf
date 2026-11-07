#include "../testing_utils.h"

#include <complex>

#include <xsf/cpu/legendre.h>

#define MDSPAN_USE_PAREN_OPERATOR 1
#include <xsf/third_party/kokkos/mdspan.hpp>

TEST_CASE("lqn complex non-singular point with real part one", "[lqn][complex][xsf_tests]") {
    using std::complex;

    constexpr int n = 1;
    const complex<double> z{1.0, 0.2};

    complex<double> q_data[n + 1];
    complex<double> qp_data[n + 1];

    auto q = std::mdspan(q_data, n + 1);
    auto qp = std::mdspan(qp_data, n + 1);

    xsf::lqn(z, q, qp);

    const complex<double> expected_q0 = std::log(-(1.0 + z) / (1.0 - z)) / 2.0;

    const complex<double> expected_q1 = z * expected_q0 - 1.0;

    const double rtol = 1e-13;

    const auto error_q0 = xsf::extended_relative_error(q(0), expected_q0);
    const auto error_q1 = xsf::extended_relative_error(q(1), expected_q1);

    CAPTURE(z, q(0), expected_q0, error_q0);
    CAPTURE(z, q(1), expected_q1, error_q1);

    REQUIRE(error_q0 <= rtol);
    REQUIRE(error_q1 <= rtol);
}

TEST_CASE("lqn complex backward recurrence near unit circle", "[lqn][complex][xsf_tests]") {
    using std::complex;

    constexpr int n = 5;
    const complex<double> z{0.0, 1.01};

    const complex<double> expected[n + 1] = {{0.0, -0.780423080066594232},  {-0.211772689132739825, 0.0},
                                             {0.0, 0.0693759159971962806},  {0.0243990008265461447, 0.0},
                                             {0.0, -0.0089067030369768997}, {-0.00332681454001291209, 0.0}};

    const complex<double> expected_derivatives[n + 1] = {{0.4950249987624375, 0.0},   {0.0, -0.28044783131653234},
                                                         {-0.14029306863578198, 0.0}, {0.0, 0.066431748669449051},
                                                         {0.03049993715004104, 0.0},  {0.0, -0.013728578663343049}};

    complex<double> q_data[n + 1];
    complex<double> qp_data[n + 1];

    auto q = std::mdspan(q_data, n + 1);
    auto qp = std::mdspan(qp_data, n + 1);

    xsf::lqn(z, q, qp);

    const double rtol = 1e-12;

    for (int k = 0; k <= n; ++k) {
        const auto value_error = xsf::extended_relative_error(q(k), expected[k]);

        const auto derivative_error = xsf::extended_relative_error(qp(k), expected_derivatives[k]);

        CAPTURE(z, k);
        CAPTURE(q(k), expected[k], value_error);
        CAPTURE(qp(k), expected_derivatives[k], derivative_error);

        CHECK(value_error <= rtol);
        CHECK(derivative_error <= rtol);
    }
}

TEST_CASE("lqn complex zero backward recurrence depth", "[lqn][complex][xsf_tests]") {
    using std::complex;

    constexpr int n = 5;
    const complex<double> z{0.95, 0.4};

    const complex<double> expected[n + 1] = {
        {0.79848810410393597, -0.746414846481677156},     {0.0571296374914100336, -0.389698862516018911},
        {-0.0840150011170973405, -0.147835673349642349},  {-0.0725530611965825785, -0.0302839085376526731},
        {-0.0364094774251386603, 0.00974261423077638805}, {-0.0112324396858800456, 0.0146721734186499267}
    };

    complex<double> q_data[n + 1]{};
    complex<double> qp_data[n + 1]{};

    auto q = std::mdspan(q_data, n + 1);
    auto qp = std::mdspan(qp_data, n + 1);

    xsf::lqn(z, q, qp);

    const double rtol = 1e-12;

    for (int k = 0; k <= n; ++k) {
        const auto error = xsf::extended_relative_error(q(k), expected[k]);

        CAPTURE(z, k, q(k), expected[k], error);
        CHECK(error <= rtol);
    }
}

TEST_CASE("lqn complex recurrence near unit circle at high degree", "[lqn][complex][xsf_tests]") {
    using std::complex;

    constexpr int n = 20;

    const complex<double> z{0.64259491521733225, 0.76633660028529871};

    const std::pair<int, complex<double>> expected[] = {
        {0, {0.38123595425448242, -0.7853329146546667}},
        {1, {-0.15319035838944023, -0.21249587261999422}},
        {5, {0.0059957721278025983, -0.0011290654783228573}},
        {10, {6.1784837449382115e-05, 5.9177031494495501e-05}},
        {20, {-2.0181412353476887e-08, 1.0363347042627469e-08}}
    };

    const std::pair<int, complex<double>> expected_derivatives[] = {
        {0, {0.49991484777418038, 0.41926413539100893}},
        {1, {0.38118124133871645, -0.13281286824400512}},
        {5, {-0.0040752032371961684, 0.02874250348099042}},
        {10, {-0.00067060784431434992, 0.0003408645281018296}},
        {20, {-5.0347989487523367e-08, -3.7935620451654274e-07}}
    };

    complex<double> q_data[n + 1]{};
    complex<double> qp_data[n + 1]{};

    auto q = std::mdspan(q_data, n + 1);
    auto qp = std::mdspan(qp_data, n + 1);

    xsf::lqn(z, q, qp);

    const double rtol = 1e-12;

    for (const auto &[k, reference] : expected) {
        const auto error = xsf::extended_relative_error(q(k), reference);

        CAPTURE(z, k, q(k), reference, error);
        CHECK(error <= rtol);
    }

    for (const auto &[k, reference] : expected_derivatives) {
        const auto error = xsf::extended_relative_error(qp(k), reference);

        CAPTURE(z, k, qp(k), reference, error);
        CHECK(error <= rtol);
    }
}

TEST_CASE("lqn complex float recurrence produces finite values", "[lqn][complex][xsf_tests]") {
    using std::complex;

    constexpr int n = 5;
    const complex<float> z{0.0f, 1.01f};

    const complex<float> expected[n + 1] = {{0.0f, -0.780423085f}, {-0.211772692f, 0.0f},   {0.0f, 0.0693759173f},
                                            {0.0243990015f, 0.0f}, {0.0f, -0.00890670333f}, {-0.00332681467f, 0.0f}};

    complex<float> q_data[n + 1]{};
    complex<float> qp_data[n + 1]{};

    auto q = std::mdspan(q_data, n + 1);
    auto qp = std::mdspan(qp_data, n + 1);

    xsf::lqn(z, q, qp);

    for (int k = 0; k <= n; ++k) {
        CAPTURE(k, q(k), qp(k));
        CHECK(std::isfinite(q(k).real()));
        CHECK(std::isfinite(q(k).imag()));
        CHECK(std::isfinite(qp(k).real()));
        CHECK(std::isfinite(qp(k).imag()));
        const auto error = xsf::extended_relative_error(
            static_cast<std::complex<double>>(q(k)), static_cast<std::complex<double>>(expected[k])
        );
        CAPTURE(k, q(k), expected[k], error);
        CHECK(error <= 1e-5);
    }
}

TEST_CASE("lqn complex recurrence across radius 1.1", "[lqn][complex][xsf_tests]") {
    using std::complex;

    constexpr int n = 20;

    using test_case = std::tuple<complex<double>, complex<double>, complex<double>, complex<double>, complex<double>>;

    auto [z, q0, q1, q5, q20] = GENERATE(
        test_case{
            {0.70678308583556659, 0.84288520545668266},
            {0.37876801438233643, -0.72343196659446607},
            {-0.1225230721821335, -0.19205152211865384},
            {0.0039433678219418832, -0.00010641402497356082},
            {-4.7939849376853453e-09, -3.1725322618201577e-10}
        },
        test_case{
            {0.70678437089689095, 0.84288673797663127},
            {0.37876792087239186, -0.72343079301348723},
            {-0.12252253205529735, -0.19205112065315535},
            {0.003943334289364729, -0.00010640154028816038},
            {-4.7938355792971064e-09, -3.1729121743884457e-10}
        }
    );

    complex<double> q_data[n + 1]{};
    complex<double> qp_data[n + 1]{};

    auto q = std::mdspan(q_data, n + 1);
    auto qp = std::mdspan(qp_data, n + 1);

    xsf::lqn(z, q, qp);

    const std::pair<int, complex<double>> expected[] = {{0, q0}, {1, q1}, {5, q5}, {20, q20}};

    // References generated with mpmath 1.4.1, dps=50,
    // using legenq(k, 0, z, type=3).
    constexpr double rtol = 1e-12;

    for (const auto &[k, reference] : expected) {
        const auto error = xsf::extended_relative_error(q(k), reference);

        CAPTURE(z, k, q(k), reference, error);
        CHECK(error <= rtol);
    }
}
