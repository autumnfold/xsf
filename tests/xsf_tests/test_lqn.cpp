#include "../testing_utils.h"

#include <complex>

#include <xsf/cpu/legendre.h>

#define MDSPAN_USE_PAREN_OPERATOR 1
#include <xsf/third_party/kokkos/mdspan.hpp>

// Reference values were generated with mpmath 1.4.1 at mp.dps = 50:
//
//     Q_k(z)  = legenq(k, 0, z, type=3)
//     Q_k'(z) = mp.diff(lambda t: legenq(k, 0, t, type=3), z)
//
// and rounded to the nearest double. Each z is converted to mpmath exactly from
// the double literal used in the test (e.g. mpc(complex(0.0, 1.0002))), not from
// its decimal spelling. For complex<float> inputs, z is first rounded to binary32
// (e.g. 1.01f == 1.0099999904632568359375) and the references are rounded to the
// nearest float.

TEST_CASE("lqn complex non-singular point with real part one", "[lqn][complex][xsf_tests]") {
    using std::complex;

    // Before the fix, any z with real(z) == 1 was treated as the singular point z = 1.
    constexpr int n = 5;
    const complex<double> z{1.0, 0.2};

    const complex<double> expected[n + 1] = {
        {1.153780129210315, -0.7355638371518672},     {0.3008928966406883, -0.5048078113098043},
        {0.025891623748816338, -0.29916192939656633}, {-0.057721915046909524, -0.15343413353813556},
        {-0.06673012240535647, -0.06434095691073081}, {-0.05077394380425094, -0.01708925967473535}
    };

    const complex<double> expected_derivatives[n + 1] = {
        {0.24752475247524752, 2.4752475247524752},  {0.9062553767350673, 1.7891886380956574},
        {1.1502034423973124, 0.9608240908230622},   {1.035713495479149, 0.29337899111282567},
        {0.7461500370689458, -0.11321484394388674}, {0.4351423938309408, -0.2856896210837516}
    };

    complex<double> q_data[n + 1]{};
    complex<double> qp_data[n + 1]{};

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

TEST_CASE("lqn complex singular endpoints return sentinel", "[lqn][complex][xsf_tests]") {
    using std::complex;

    // z = +1 and z = -1 both return the 1e300 sentinel, matching the real lqn.
    // Previously only z = +1 was detected; z = -1 produced non-finite values.
    const complex<double> z = GENERATE(complex<double>{1.0, 0.0}, complex<double>{-1.0, 0.0});

    constexpr int n = 3;

    complex<double> q_data[n + 1]{};
    complex<double> qp_data[n + 1]{};

    auto q = std::mdspan(q_data, n + 1);
    auto qp = std::mdspan(qp_data, n + 1);

    xsf::lqn(z, q, qp);

    const complex<double> sentinel{1e300, 0.0};

    for (int k = 0; k <= n; ++k) {
        CAPTURE(z, k, q(k), qp(k));
        CHECK(q(k) == sentinel);
        CHECK(qp(k) == sentinel);
    }
}

TEST_CASE("lqn complex negative real axis beyond -1", "[lqn][complex][xsf_tests]") {
    using std::complex;

    // The old recurrence-depth heuristic used abs(z - 1), giving a negative depth here.
    constexpr int n = 5;
    const complex<double> z{-1.05, 0.0};

    const complex<double> expected[n + 1] = {{-1.8567860333521535, 0.0},  {0.9496253350197612, 0.0},
                                             {-0.5672668859800473, 0.0},  {0.35963349378524195, 0.0},
                                             {-0.23537638034534664, 0.0}, {0.1571545638245116, 0.0}};

    const complex<double> expected_derivatives[n + 1] = {{-9.7560975609756, 0.0},   {8.387116405672227, 0.0},
                                                         {-6.907221555916317, 0.0}, {5.550781975771992, 0.0},
                                                         {-4.389787099419624, 0.0}, {3.432394552663872, 0.0}};

    complex<double> q_data[n + 1]{};
    complex<double> qp_data[n + 1]{};

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

TEST_CASE("lqn complex backward recurrence rescaling at high degree", "[lqn][complex][xsf_tests]") {
    using std::complex;

    // |z| - 1 = 2e-4 gives a recurrence depth of 840, over which the unscaled
    // backward recurrence grows past the double range.
    constexpr int n = 20;

    const complex<double> z{0.0, 1.0002};

    const std::pair<int, complex<double>> expected[] = {
        {0, {0.0, -0.7852981733967817}},
        {1, {-0.214544766968539, 0.0}},
        {5, {-0.003464436601942523, 0.0}},
        {10, {0.0, 3.0805408607313336e-05}},
        {20, {0.0, -3.287301721857672e-09}}
    };

    const std::pair<int, complex<double>> expected_derivatives[] = {
        {0, {0.4999000099999998, 0.0}},
        {1, {0.0, -0.28529818339478186}},
        {5, {0.0, -0.014362758868771768}},
        {10, {-0.00023651815832093483, 0.0}},
        {20, {4.8475608240001165e-08, 0.0}}
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

TEST_CASE("lqn complex float recurrence is finite and accurate", "[lqn][complex][xsf_tests]") {
    using std::complex;

    constexpr int n = 5;
    const complex<float> z{0.0f, 1.01f};

    const complex<float> expected[n + 1] = {{0.0f, -0.780423105f}, {-0.211772695f, 0.0f},   {0.0f, 0.0693759173f},
                                            {0.0243990012f, 0.0f}, {0.0f, -0.00890670344f}, {-0.00332681462f, 0.0f}};

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

    constexpr double rtol = 1e-12;

    for (const auto &[k, reference] : expected) {
        const auto error = xsf::extended_relative_error(q(k), reference);

        CAPTURE(z, k, q(k), reference, error);
        CHECK(error <= rtol);
    }
}
