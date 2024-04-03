#include "UnitTest.h"
#include "DeltaHedgerReal.h"

// unit tests to check implied vol calculation and delta calculation
void UnitTest::Test_implied_volatility() {
    DeltaHedgerReal HedgerReal(500, "2011-07-05", "2011-07-29", "2011-09-17");
    vector <double> implied_volitility = HedgerReal.implied_volitility_copy;
    cout << "Now Begin to Test Implied Volatility!" << endl;
    cout << "Input Data: Strike Price: 500, Start Date: 2011-07-05, End Date: 2011-07-29, Maturity Date: 2011-09-17" << endl;
    cout << "Implied Volatility:" << endl;
    for (int i = 0 ; i < implied_volitility.size();i++ )
    {
        cout << implied_volitility[i] << " ";
    }
    cout << "" << endl;
    cout << "Successfully Test Implied Volatility!" << endl;
};

void UnitTest::Test_dalta() {
    DeltaHedgerReal HedgerReal(500, "2011-07-05", "2011-07-29", "2011-09-17");
    vector <double> delta = HedgerReal.delta_vector_copy;
    cout << "Now Begin to Test Delta!" << endl;
    cout << "Input Data: Strike Price: 500, Start Date: 2011-07-05, End Date: 2011-07-29, Maturity Date: 2011-09-17" << endl;
    cout << "Delta:" << endl;
    for (int j = 0 ; j < delta.size(); j++ )
    {
        cout << delta[j] << " ";
    }
    cout << "" << endl;
    cout << "Successfully Test Delta!" << endl;
};
