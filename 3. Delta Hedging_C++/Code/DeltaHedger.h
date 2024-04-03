#ifndef DeltaHedger_h
#define DeltaHedger_h

#include <string>
#include <vector>
using namespace std;

class DeltaHedger {
private:
    double K, S, T, r, mu, sigma, N; string flag;
    void init();

public:

    DeltaHedger();

    DeltaHedger(double underlying_asset, double strike_price, double TTM, double mean, double volatility, double risk_free_rate,
                double n_interval, string flag_PC);

    ~DeltaHedger();

    static pair<double, double> BSM_Pricer(double underlying_asset, double strike_price, double TTM, double volatility, double risk_free_rate, string flag_PC);

    vector<double> Stock_Simulate() const;
    vector<double> Option_Simulate(vector<double> stock_price_vector);
    vector<double> HedgingError(vector<double> stock_price_vector, vector<double> option_price_vector);

    void Simulate(int simulation_times);
};

#endif
