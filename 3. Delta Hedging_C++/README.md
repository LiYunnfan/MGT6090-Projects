# HW3 Project Description

## Function Explanation

### `StdNormalCDF.h` / `StdNormalCDF.cpp`
- **`standardNormalCDF()` Function:**
  - Calculate the Cumulative Distribution Function (CDF) of a Normal Distribution.

### `Option.h`/ `Option.cpp`
- **`Get_K(), Get_S(), ...` Functions:**
  - Retrieve the values of `K`, `S`, `r`, `T`, `sigma`.
- **`init()` Function:**
  - Initialize the values of `K`, `S`, `r`, `T`, `sigma`.

### `DeltaHedger.h` / `DeltaHedger.cpp`
- **`BSM_Pricer()` Function:**
  - Calculate Black-Scholes Price for Option.
- **`Stock_Simulate()` Function:**
  - Simulate Stock Price with N.
- **`Option_Simulate()` Function:**
  - Simulate Option Price with N.
- **`HedgingError()` Function:**
  - Calculate Hedge Error with formula in PDF.
- **`Simulate()` Function:**
  - Simulate Stock Price, Option Price and Hedge Error; Write them into csv.

### `DeltaHedgerReal.h` / `DeltaHedgerReal.cpp`
- **`read_interest_csv(), read_stock_csv(), read_option_csv()` Function:**
  - Read Interest/StockPrice/OptionPrice.csv and put them into vectors.
- **`find_data()` Function:**
  - Find data corresponding a specific date.
- **`calculate_implied_volatility(), calculate_Delta(), calculate_PNL_Hedge(), calculate_PNL()` Function:**
  - Calculate Implied Volatility, Delta, PNL With Hedge and PNL.

### `UnitTest.h` / `UnitTest.cpp`
- **`Test_implied_volatility()/Test_dalta` Function:**
  - Test Calculated Implied Volatility and Delta.

### `main()` Function
1. **Task 1:** Simulation with 1000 times.
2. **Task 2:** Construct The Delta-hedging Portfolio For GOOG.

## Program Run Guidance

### For MAC Users:
1. Use the `cd` command to navigate to the directory containing the file.
2. Run the executable file using the `./main` command.

### For IDE Users:
- Ensure the working path is the same as the file path to run the program successfully.

### Compilation Instruction:
- Use the following command to compile the files:
  ```bash
  g++ DeltaHedger.cpp DeltaHedgerReal.cpp main.cpp Option.cpp StdNormalCDF.cpp UnitTest.cpp -o main
