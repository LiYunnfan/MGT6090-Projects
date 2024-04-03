#include "DeltaHedger.h"
#include "DeltaHedgerReal.h"
#include "UnitTest.h"

int main()
{   
    
    // Task 1: Simulate 1000 Time for Stock Price, Option Price and Hedge Error.
    DeltaHedger Hedger;
    Hedger.Simulate(1000);
    cout << "Task 1: Simulate 1000 Time for Stock Price, Option Price and Hedge Error Completed." << endl;

    // Task 2: Construct The Delta-hedging Portfolio For GOOG.
    double strick_price = 500;
    string T = "2011-09-17";
    string t_0 = "2011-07-05";
    string t_N = "2011-07-29";
    DeltaHedgerReal RealTrade(strick_price,t_0,t_N,T);
    cout << "Task 2: Construct The Delta-hedging Portfolio For GOOG Completed." << endl;

    // UnitTest: Test Implied Volatility and Delta.
    UnitTest unit_test;

    unit_test.Test_implied_volatility();
    unit_test.Test_dalta();

    return 0;
}