#ifndef UNITTEST_h
#define UNITTEST_h

#include "Option.h"
#include "DeltaHedgerReal.h"
#include <vector>

class UnitTest {
public:
    // unit tests to check implied vol calculation and delta calculation
    void Test_implied_volatility();
    void Test_dalta();
};


#endif