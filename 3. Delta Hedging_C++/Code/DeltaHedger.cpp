#include "DeltaHedger.h"
#include "StdNormalCDF.h"
#include <fstream>
#include <string>
#include <cmath>
#include <random>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;

void DeltaHedger::init() {
        S = 100;
        K = 105;
        T = 0.4;
        mu = 0.05;
        sigma = 0.24;
        r = 0.025;
        N = 100;
        flag = "C";
    };

DeltaHedger::DeltaHedger(){init();}

DeltaHedger::DeltaHedger(double underlying_asset, double strike_price, double TTM, double mean, double volatility, double risk_free_rate,
                double n_interval, string flag_PC) {
    S = underlying_asset;
    K = strike_price;
    T = TTM;
    mu = mean;
    sigma = volatility;
    r = risk_free_rate;
    N = n_interval;
    flag =flag_PC;
}

DeltaHedger::~DeltaHedger(){};

pair<double, double> DeltaHedger::BSM_Pricer(double underlying_asset, double strike_price, double TTM, double volatility, double risk_free_rate, string flag_PC) {

    double q = 0;

    pair<double, double> result;

    if (flag_PC == "C" or flag_PC == "c")
    {
        StdNormalCDF CDF_method;
        double d1 = (log(underlying_asset / strike_price) + (risk_free_rate + 0.5 * volatility * volatility - q) * TTM) / (volatility * sqrt(TTM));
        double d2 = d1 - volatility * sqrt(TTM);
        double V_call = underlying_asset * exp(-q * TTM) * CDF_method.CDF(d1) - strike_price * exp(-risk_free_rate * TTM) * CDF_method.CDF(d2);
        double delta_call = exp(-q*TTM)*CDF_method.CDF(d1);
        result.first = V_call;
        result.second = delta_call;
        return result;
    };

     if (flag_PC == "P" or flag_PC == "p")
    {
        StdNormalCDF CDF_method;
        double d1 = (log(underlying_asset / strike_price) + (risk_free_rate + 0.5 * volatility * volatility - q) * TTM) / (volatility * sqrt(TTM));
        double d2 = d1 - volatility * sqrt(TTM);
        double V_put = -underlying_asset * exp(-q * TTM) * CDF_method.CDF(-d1) + strike_price * exp(-risk_free_rate * TTM) * CDF_method.CDF(-d2);
        double delta_put = exp(-q*TTM)*(CDF_method.CDF(d1)-1);
        result.first = V_put;
        result.second = delta_put;
        return result;
    };
};

// Simulate Stock Price
vector<double> DeltaHedger::Stock_Simulate() const {

    vector<double> stock_price_vector(N + 1);
    random_device rand;
    mt19937 gen(rand());
    normal_distribution<float> StdNorm(0, 1);

    // default_random_engine generator;
    // normal_distribution<double> distribution(0.0, 1.0);
    stock_price_vector[0] = S;
    double dt = T / N;
    for (int i = 1; i <= N; i++) {
        double Z = StdNorm(gen);

        stock_price_vector[i] = stock_price_vector[i - 1] * (1 + mu * dt + sigma * sqrt(dt) * Z);
        }
    
    return stock_price_vector;
}

// Simulate Option Price
vector<double> DeltaHedger::Option_Simulate(vector<double> stock_price_vector) {

    vector<double> option_price_vector(N + 1);
    pair<double, double> option_price_delta;
    double dt = T / N;

    for (int i = 0; i <= N; i++) {
         double T_period = T - i * dt; // Notice to change the maturity time
         option_price_delta = BSM_Pricer(stock_price_vector[i], K, T_period, sigma, r, flag);
         option_price_vector[i] = option_price_delta.first;
    }
    return option_price_vector;
}

// Calculate Hedging Error
vector<double> DeltaHedger::HedgingError(vector<double> stock_price_vector, vector<double> option_price_vector) {

    vector<double> delta_vector(N + 1), B_vector(N + 1), HE_vector(N + 1);
    pair<double, double> option_price_delta;
    double dt = T / N;

    for (int i = 0; i <= N; i++) {
         double T_period = T - i * dt;
         option_price_delta = BSM_Pricer(stock_price_vector[i], K, T_period, sigma, r, flag);
         delta_vector[i] = option_price_delta.second;
    }

    for (int i = 0; i <= N; i++) {
        if (i == 0) {
            B_vector[i] = option_price_vector[i] - delta_vector[i] * stock_price_vector[i];
        }
        else {
            B_vector[i] = (delta_vector[i - 1] - delta_vector[i]) * stock_price_vector[i] + B_vector[i - 1] * exp(r * dt);
        }
    }

    for (int i = 0; i <= N; i++) {
        if (i == 0) {
            HE_vector[i] = 0;
        }
        else {
            HE_vector[i] = delta_vector[i - 1] * stock_price_vector[i] + B_vector[i - 1] * exp(r * dt) - option_price_vector[i];
        }
    }

    return HE_vector;
}

// Compile them into Simulate
void DeltaHedger::Simulate(int simulation_times) {

    vector<vector<double> >  stock_vector(simulation_times), option_vector(simulation_times);
    vector<double>  HE_series(simulation_times);

    for (int i = 0; i < simulation_times; i++) {
        stock_vector[i] = Stock_Simulate();
        option_vector[i] = Option_Simulate(stock_vector[i]);
        HE_series[i] = HedgingError(stock_vector[i], option_vector[i])[N];
    }

    ofstream stock_csv, option_csv, HE_csv;

    stock_csv.open("./Stock_Simulate.csv");
    for (int i = 0; i < simulation_times; i++) {
        for (int j = 0; j <= N; j++) {
            stock_csv << stock_vector[i][j] << ",";
        }
        stock_csv << endl;
    }
    stock_csv.close();

    option_csv.open("./Option_Simulate.csv");
    for (int i = 0; i < simulation_times; i++) {
        for (int j = 0; j <= N; j++) {
            option_csv << option_vector[i][j] << ",";
        }
        option_csv << endl;
    }
    option_csv.close();

    HE_csv.open("./HE_Simulate.csv");
    for (int i = 0; i < simulation_times; i++) {
        HE_csv << HE_series[i] << endl;
    }
    HE_csv.close();
}