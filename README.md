# Insurance Risk Analysis - Cramér-Lundberg Model

Stochastic simulation of the Cramér-Lundberg model for insurance risk analysis. Main goals were to test and visualize the effect of initial capital and premium on the probability of ruin of the insurance company. Furthermore, the distributions of claims' arrival time and/or size are changed to understand their differences and effect on the results. Finally, three approximations are tested.

## Problem Setup

Analysis based on Danish insurance company data:
- **Claim sizes**: Uniform distribution (mean: 200k DKK, std: 50k DKK)
- **Claims arrival**: Poisson process (5 claims/day)  
- **Model**: U(t) = u₀ + c·t - S(t)

## Features

### Monte Carlo Simulation
- Ruin probability estimation for different initial capital and premium combinations
- Deficit at ruin and recovery time analysis

### Distribution Analysis  
- **Claim sizes**: Uniform vs Erlang-2
- **Inter-arrival times**: Exponential, Gamma, Erlang, Uniform
- Impact of arrival time variance on ruin probability

### Approximation Methods
Three analytical approximations for ruin probability ψ(u):
1. Exponential: ψ(u) ≈ ψ(0)e^(-Ru)
2. Moment-based: ψ(u) ≈ ψ(0)e^(-ku)
3. Gamma distribution approximation

## Key Results

- Ruin probability decreases with initial capital and premium rate
- Higher variance in inter-arrival times increases ruin probability
- Approximations (i) and (ii) more accurate than (iii)
- Recovery time negatively correlated with premium rate

## Dependencies

```python
numpy, pandas, matplotlib, scipy, statsmodels
```

---
*Academic project -Course: 2DF30 Insurance and Credit Risk, Eindhoven University of Technology *
