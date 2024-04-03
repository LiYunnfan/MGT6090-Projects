# Dynamic Delta Hedging Strategy in C++

## Overview
This repository houses a C++ implementation of a dynamic delta hedging strategy, which seeks to replicate the financial performance of a derivative position by dynamically adjusting holdings in the underlying asset. The codebase utilizes the Black-Scholes model for option pricing and to compute the Greeks for hedging purposes.

## Repository Structure
- `DeltaHedger.cpp` & `DeltaHedger.h`: Core algorithm for delta hedging strategy.
- `DeltaHedgerReal.cpp` & `DeltaHedgerReal.h`: Adaptation of the hedging strategy to real market data.
- `main.cpp`: Entry point for the simulations and real market analysis.
- `Option.cpp` & `Option.h`: Option class for representing options with pricing functionality.
- `StdNormalCDF.cpp` & `StdNormalCDF.h`: Standard normal cumulative distribution function used in option pricing.
- `UnitTest.cpp` & `UnitTest.h`: Unit tests for validating the correctness of the implementation.

## Building the Project
Ensure you have a C++ compiler and make utility installed. Then run:

\```bash
make main
\```

This will compile the source files and link the binaries to create an executable named `main`.

## Running the Simulation
After building the project, run the simulation by executing the compiled binary:

\```bash
./main
\```

## Testing
To run unit tests, ensure you have a unit testing framework such as Google Test installed, then compile and run the tests:

\```bash
make test
\```

