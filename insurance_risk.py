import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy import cumsum, zeros, mean
from math import sqrt
from statsmodels.stats.weightstats import DescrStatsW


def multiplyByFactors(factors, nr):
    """
    Multiplies a number by an array of multipliers and returns an array
    of the products
    @param factors: array of multipliers
    @param nr: number to be multiplied
    @return: array containing the product of the number and each
    modifier
    """
    res = np.zeros(len(factors))
    for i, fac in enumerate(factors):
        res[i] = nr * fac
    return res


def createModel(u: int, c: int, params: dict):
    """
    Returns data about the capital model of the insurance company.
    @param u: initial capital
    @param c: premium per time unit
    @param params: extra parameters containing distribution types
    @return: array containing arrivalTimes sample, cumulative claim
    size, U(t) and ruin probability check
    """
    lam = 5  # Poisson rate: 5 claims per day
    t = 1000  # Time horizon: 1000 days
    N = round(lam * t)  # Expected number of claims

    # Extract distribution parameters
    claimSizeDistribution = params.get("claim_sizes_distribution")
    interArrivalTimeDistribution = params.get("arrival_time_distribution")

    # Generate claim sizes based on distribution type
    if claimSizeDistribution == "U":
        # Uniform distribution with mean=200000, std=50000
        # a ≈ 113397, b ≈ 286603
        claimSizes = stats.uniform(loc=113397, scale=173206).rvs(N)
    elif claimSizeDistribution == "G":
        # Gamma distribution (placeholder implementation)
        claimSizes = stats.gamma(1).rvs(N)

    cumulativeClaimSizes = cumsum(claimSizes)

    # Generate inter-arrival times based on distribution type
    if interArrivalTimeDistribution == "E":
        # Exponential distribution (Poisson process)
        interArrivals = stats.expon(scale=1 / lam).rvs(N)
    elif interArrivalTimeDistribution == "G":
        # Gamma distribution
        alpha = params.get("alpha")
        beta = params.get("beta")
        interArrivals = stats.gamma(a=alpha, scale=1 / beta).rvs(N)
    elif interArrivalTimeDistribution == "U":
        # Uniform distribution
        a = params.get("a")
        b = params.get("b")
        interArrivals = stats.uniform(loc=a, scale=b - a).rvs(N)
    elif interArrivalTimeDistribution == "E4":
        # Erlang-4 distribution
        interArrivals = stats.gamma(a=4, scale=1 / 20).rvs(N)

    arrivalTimes = cumsum(interArrivals)

    # Cramér-Lundberg model: U(t) = u + c*t - S(t)
    capital = u + arrivalTimes * c - cumulativeClaimSizes
    ruin = min(capital) < 0

    return arrivalTimes, cumulativeClaimSizes, capital, ruin


def simModel(nrRuns: int, u: int, c: int, params: dict) -> tuple:
    """
    Simulates a model a number of times and calculates ruin probability,
    deficit at ruin, and recovery time.
    @param nrRuns: number of runs for simulating the model
    @param u: initial capital
    @param c: premium per time unit
    @param params: extra parameters
    @return: ruin probability, mean of deficit at ruin and mean
    of recovery time
    """
    ruin_list = zeros(nrRuns)
    deficit = []
    recovery_time = []

    for run in range(nrRuns):
        arrivalTimes, cumulativeClaimSizes, capital, ruin = createModel(u, c, params)
        ruin_list[run] = ruin

        # Track deficit and recovery time
        negative = False
        time_of_ruin = 0

        for i in range(len(arrivalTimes)):
            if capital[i] <= 0 and not negative:
                # Ruin occurs
                negative = True
                deficit.append(abs(capital[i]))
                time_of_ruin = arrivalTimes[i]
            elif capital[i] > 0 and negative:
                # Recovery occurs
                negative = False
                recovery_time.append(arrivalTimes[i] - time_of_ruin)

        # Handle case where capital doesn't recover by end of simulation
        if ruin and len(recovery_time) == 0:
            recovery_time.append(1000)  # Use max time as placeholder

    # Handle edge cases for means
    if not deficit:
        deficit = [0]
    if not recovery_time:
        recovery_time = [0]

    return mean(ruin_list), mean(deficit), mean(recovery_time)


def approxUniform1(u: int, c: int, R: float) -> float:
    """
    Result of the i) approximation formula: ψ(u) ≈ ψ(0) * e^(-Ru)
    @param u: initial capital
    @param c: premium per time unit
    @param R: adjustment coefficient
    @return: approximation value for i)
    """
    psi_0 = (10 ** 6) / c  # ψ(0) = 1/(1+θ) where θ = c/(λμ) - 1
    formula = psi_0 * np.exp(-R * u)
    return formula


def approxUniform2(u: int, c: int) -> float:
    """
    Result of the ii) approximation formula using moments.
    @param u: initial capital
    @param c: premium per time unit
    @return: approximation value for ii)
    """
    theta = (10 ** (-6)) * c - 1
    psi_0 = 1 / (1 + theta)
    exponent = (-2 * theta * 200000) / ((1 + theta) * 4.25e10) * u
    formula = psi_0 * np.exp(exponent)
    return formula


def approxUniform3(u: int, c: int) -> float:
    """
    Result of the iii) approximation formula using Gamma distribution.
    @param u: initial capital
    @param c: premium per time unit
    @return: approximation value for iii)
    """
    theta = (10 ** (-6)) * c - 1
    psi_0 = 1 / (1 + theta)

    # Calculate Gamma parameters
    beta = ((0.6025 + 2.258 / theta) / (106250 * 10 ** (-10)) - c * 10 ** (-6)) ** (-1)
    alpha = (0.106250 * beta * c) / theta

    # Use survival function: 1 - CDF
    formula = psi_0 * stats.gamma.sf(x=u, a=alpha, scale=1 / beta)
    return formula


def psi_E2(theta: float, u: int) -> float:
    """
    Expression for Erlang-2 distributed claim sizes ruin probability.
    @param theta: safety factor θ = c/(λμ) - 1
    @param u: initial capital
    @return: approximation value for Erlang-2 distributed claim sizes
    """
    sqrt_term = sqrt(8 * theta + 9)

    c1 = (-5 - 4 * theta + sqrt_term) / (3 + 4 * theta + sqrt_term)
    c2 = (5 + 4 * theta + sqrt_term) / (3 + 4 * theta - sqrt_term)

    alpha1 = (3 + 4 * theta + sqrt_term) / (2 * (1 + theta) * 200000)
    alpha2 = (3 + 4 * theta - sqrt_term) / (2 * (1 + theta) * 200000)

    psi = (theta / ((1 + theta) * sqrt_term)) * (
            c1 * math.exp(-alpha1 * u) + c2 * math.exp(-alpha2 * u)
    )
    return psi


def monteCarloSim(premiumInterval: list, initialCapitalInterval: list, params: dict):
    """
    Returns a matrix of all possible combinations of premiums and initial capitals
    @param premiumInterval: the interval of the possible values of
    the amount of premium per time unit
    @param initialCapitalInterval: the interval of the possible values
    of the initial capital
    @param params: extra parameters
    @return: matrix of all possible combinations
    """
    result = []
    for iP, premium in enumerate(premiumInterval):
        iteration = []
        for iC, capital in enumerate(initialCapitalInterval):
            # Run simulation
            sim_results = simModel(params.get("nrRuns"), capital, premium, params)

            # Calculate approximations
            obj = {
                "ruin": sim_results[0],
                "deficit": sim_results[1],
                "recovery_time": sim_results[2],
                "approx1": approxUniform1(capital, premium, params.get("valuesOfR")[iP]),
                "approx2": approxUniform2(capital, premium),
                "approx3": approxUniform3(capital, premium),
            }

            # Add Erlang-2 approximation if theta is valid
            theta = (premium * 10 ** (-6)) - 1
            if theta > 0:
                obj["erlang2"] = psi_E2(theta, capital)

            iteration.append(obj)
        result.append(np.array(iteration))

    # Create CSV if filepath provided
    if params.get("filepath"):
        createCSV(result, premiumInterval, initialCapitalInterval, params.get("filepath"))

    return result


def monteCarloSimPrint(premiumInterval: list, initialCapitalInterval: list, params: dict):
    """
    Prints a matrix of the ruin probabilities of the Monte Carlo simulation
    @param premiumInterval: premium values to test
    @param initialCapitalInterval: initial capital values to test
    @param params: simulation parameters
    """
    results = monteCarloSim(premiumInterval, initialCapitalInterval, params)

    print("Premium--Capital: Simulation | Approx1 | Approx2 | Approx3")
    print("-" * 60)

    for i, premium in enumerate(premiumInterval):
        print(f"\nPremium = {premium}")
        for j, capital in enumerate(initialCapitalInterval):
            result = results[i][j]
            print(f"Capital {capital}: {result['ruin']:.4f} | "
                  f"{result['approx1']:.4f} | {result['approx2']:.4f} | "
                  f"{result['approx3']:.4f}")


def createCSV(data, premiums, capitals, filepath):
    """
    Creates a CSV file showcasing a matrix of the ruin probabilities
    @param data: ruin probabilities matrix
    @param premiums: premium values
    @param capitals: capital values
    @param filepath: output file path
    """
    # Extract ruin probabilities for DataFrame
    ruin_matrix = []
    for premium_data in data:
        row = []
        for capital_data in premium_data:
            row.append(capital_data['ruin'])
        ruin_matrix.append(row)

    df = pd.DataFrame(ruin_matrix, index=premiums, columns=capitals)
    df.to_csv(filepath)
    print(f"Results saved to {filepath}")


def deficit_and_recovery_analysis(premiumInterval: list, initialCapitalInterval: list, params: dict):
    """
    Analyze deficit at ruin and recovery time statistics
    @param premiumInterval: premium values to test
    @param initialCapitalInterval: capital values to test
    @param params: simulation parameters
    """
    results = monteCarloSim(premiumInterval, initialCapitalInterval, params)

    print("Deficit at Ruin and Recovery Time Analysis")
    print("=" * 50)

    for i, premium in enumerate(premiumInterval):
        print(f"\nPremium = {premium}")
        for j, capital in enumerate(initialCapitalInterval):
            result = results[i][j]
            print(f"Capital {capital}:")
            print(f"  Ruin Probability: {result['ruin']:.4f}")
            print(f"  Mean Deficit at Ruin: {result['deficit']:.2f}")
            print(f"  Mean Recovery Time: {result['recovery_time']:.4f}")


def plot_ruin_probability_comparison(premiumInterval: list, initialCapitalInterval: list, params: dict):
    """
    Plot ruin probability comparisons between simulation and approximations
    Similar to the plots shown in the original report
    """
    results = monteCarloSim(premiumInterval, initialCapitalInterval, params)

    # Create subplots for different premium values (stacked vertically)
    fig, axes = plt.subplots(len(premiumInterval), 1, figsize=(10, 8))
    if len(premiumInterval) == 1:
        axes = [axes]

    for i, premium in enumerate(premiumInterval):
        ax = axes[i]

        # Extract data for plotting
        capitals = np.array(initialCapitalInterval) / 1e6  # Convert to millions for readability
        sim_ruin = [results[i][j]['ruin'] for j in range(len(initialCapitalInterval))]
        approx1 = [results[i][j]['approx1'] for j in range(len(initialCapitalInterval))]
        approx2 = [results[i][j]['approx2'] for j in range(len(initialCapitalInterval))]
        approx3 = [results[i][j]['approx3'] for j in range(len(initialCapitalInterval))]

        # Plot the data
        ax.scatter(capitals, sim_ruin, color='blue', label='Sim ruin', s=60, alpha=0.8)
        ax.scatter(capitals, approx1, color='green', label='Approx (i)', s=60, alpha=0.8)
        ax.scatter(capitals, approx2, color='red', label='Approx (ii)', s=60, alpha=0.8)
        ax.scatter(capitals, approx3, color='purple', label='Approx (iii)', s=60, alpha=0.8)

        # Formatting
        ax.set_xlabel('Initial capital (in DKK)', fontsize=10)
        ax.set_ylabel('Ruin probability', fontsize=10)
        ax.set_title(f'Ruin probability for c = {premium} DKK', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(max(sim_ruin), max(approx1), max(approx2), max(approx3)) * 1.1)

        # Format x-axis to show values in scientific notation if needed
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig('ruin_probability_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def compare_arrival_distributions(premiumInterval: list, initialCapitalInterval: list):
    """
    Compare ruin probabilities for different inter-arrival time distributions
    """
    distributions = {
        "Exponential": {"arrival_time_distribution": "E"},
        "Gamma(0.16, 0.8)": {"arrival_time_distribution": "G", "alpha": 0.16, "beta": 0.8},
        "Erlang(4, 20)": {"arrival_time_distribution": "E4"},
        "Uniform(0.01, 0.39)": {"arrival_time_distribution": "U", "a": 0.01, "b": 0.39}
    }

    print("Comparison of Inter-arrival Time Distributions")
    print("=" * 55)

    for dist_name, dist_params in distributions.items():
        print(f"\n{dist_name} Distribution:")

        params = {
            "claim_sizes_distribution": "U",
            "nrRuns": 1000,
            "valuesOfR": [8.8034e-7, 1.6565e-6, 4.0606e-6, 4.9828e-6, 5.7814e-6],
            **dist_params
        }

        results = monteCarloSim(premiumInterval[:2], initialCapitalInterval[:2], params)

        for i, premium in enumerate(premiumInterval[:2]):
            for j, capital in enumerate(initialCapitalInterval[:2]):
                result = results[i][j]
                print(f"  Premium {premium}, Capital {capital}: {result['ruin']:.4f}")


# Main execution code
if __name__ == "__main__":
    # Define parameter ranges
    premiumInterval = multiplyByFactors([5.5, 6, 8, 9, 10], 200000)  # [1.1M, 1.2M, 1.6M, 1.8M, 2.0M]
    initialCapitalInterval = multiplyByFactors([1, 2.5, 5, 7.5, 10], 200000)  # [0.2M, 0.5M, 1.0M, 1.5M, 2.0M]

    # Adjustment coefficients R for approximation 1
    values_of_R = [8.8034e-7, 1.6565e-6, 4.0606e-6, 4.9828e-6, 5.7814e-6]

    # Base simulation parameters
    base_params = {
        "claim_sizes_distribution": "U",  # Uniform claim sizes
        "arrival_time_distribution": "E",  # Exponential (Poisson process)
        "filepath": "./simulation_results.csv",
        "nrRuns": 100,  # Number of Monte Carlo runs
        "valuesOfR": values_of_R
    }

    print("Insurance Company Risk Analysis")
    print("==============================")

    # 1. Basic Monte Carlo simulation with approximations
    print("\n1. Running Monte Carlo simulation with approximations...")
    monteCarloSimPrint(premiumInterval, initialCapitalInterval, base_params)

    # 2. Plot ruin probability comparisons
    print("\n2. Creating ruin probability comparison plots...")
    # Use subset of parameters for clearer plots (like in the original report)
    plot_premiums = [1100000, 1200000]  # Two premium levels for comparison
    plot_ruin_probability_comparison(plot_premiums, initialCapitalInterval, base_params)

    # 3. Deficit and recovery time analysis
    print("\n3. Analyzing deficit at ruin and recovery time...")
    deficit_and_recovery_analysis(premiumInterval, initialCapitalInterval, base_params)

    # 4. Compare different inter-arrival time distributions
    print("\n4. Comparing different inter-arrival time distributions...")
    compare_arrival_distributions(premiumInterval, initialCapitalInterval)

    # 5. Erlang-2 claim size analysis
    print("\n5. Erlang-2 claim size distribution analysis...")
    for i, premium in enumerate(premiumInterval):
        theta = (premium * 1e-6) - 1
        if theta > 0:
            print(f"Premium {premium}: θ = {theta:.6f}")
            for capital in initialCapitalInterval:
                erlang2_result = psi_E2(theta, capital)
                print(f"  Capital {capital}: Erlang-2 ψ(u) = {erlang2_result:.4f}")