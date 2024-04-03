#ifndef DeltaHedgerReal_H
#define DeltaHedgerReal_H

#include <iostream>
#include <string>
#include <vector>
#include "Option.h"

using namespace std;

class DeltaHedgerReal {
private:
    double K;
    string t_0, t_N, T;

public:
    DeltaHedgerReal();
    DeltaHedgerReal(double strick_price, string t0, string tN, string T_date);
    ~DeltaHedgerReal();

    pair<vector<string>, vector<double> > read_interest_csv();
    pair<vector<string>, vector<double> > read_stock_csv();
    vector<Option> read_option_csv();

    double find_data(vector<double> rate_vector, vector<string> date_vector, string date);

    vector<Option> Option_0_N_copy;
    vector<double> implied_volitility_copy;
    vector<double> delta_vector_copy;
    vector<double> PNL_with_hedge;
    vector<double> PNL_copy;

    vector<double> calculate_implied_volatility();
    vector<double> calculate_Delta();
    vector<double> calculate_PNL_Hedge();
    vector<double> calculate_PNL();

    void write_csv();
};

#endif
