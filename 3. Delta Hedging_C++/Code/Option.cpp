#include "Option.h"
#include "StdNormalCDF.h"
#include <string>
#include <iostream>
#include <cmath>
#include <chrono>
#include <sstream>
using namespace std;

void Option::init(double Strike_price, double underlying_asset, double Option_price, double risk_free_rate,
    string flag_PC, string t0, string tN){

    K = Strike_price;
    S = underlying_asset;
    option_price = Option_price;
    r = risk_free_rate;
    flag = flag_PC;
    t_0 = t0;
    t_N = tN;

};

Option::Option(){};

Option::Option(double Strike_price, double underlying_asset, double Option_price, double risk_free_rate,
    string flag_PC, string t0, string tN) {

    init(Strike_price, underlying_asset, Option_price, risk_free_rate, flag_PC, t0, tN);

};

Option::~Option(){};

// Define get method
double Option::get_K() const {return K;};
double Option::get_S() const {return S;};
double Option::get_r() const {return r;};
double Option::get_option_price() const {return option_price;};
string Option::get_t_0() const {return t_0;};
string Option::get_t_N() const {return t_N;};
string Option::get_flag() const {return flag;};

// Calculate Business Day
int Option::countBusinessDays(struct tm start_tm, struct tm end_tm) {
    int businessDays = 0;
    while (mktime(&start_tm) <= mktime(&end_tm)) {
        if (start_tm.tm_wday != 0 && start_tm.tm_wday != 6) { 
            businessDays++;
        }
        start_tm.tm_mday++; 
        mktime(&start_tm); 
    }
    return businessDays;
}

double Option::get_T(string start_date, string end_date) {
    struct tm start_tm = {}, end_tm = {};

    strptime(start_date.c_str(), "%Y-%m-%d", &start_tm);
    strptime(end_date.c_str(), "%Y-%m-%d", &end_tm);

    int businessDays = countBusinessDays(start_tm, end_tm);
    return static_cast<double>(businessDays);
}

// BSM pricer
pair<double, double> Option::BSM_Pricer(double sigma) 
{
    double K = Option::get_K();
    double S = Option::get_S();
    double r = Option::get_r();
    string t_0 = Option::get_t_0();
    string t_N = Option::get_t_N();
    string flag = Option::get_flag();
    double T = Option::get_T(t_0,t_N)/252;
    double q = 0;

    StdNormalCDF CDF_method;

    pair<double, double> result;

    if (flag == "C" or flag == "c")
    {
        double d1 = (log(S / K) + (r + 0.5 * sigma * sigma - q) * T) / (sigma * sqrt(T));
        double d2 = d1 - sigma * sqrt(T);
        double V_call = S * exp(-q * T) * CDF_method.CDF(d1) - K * exp(-r * T) * CDF_method.CDF(d2);
        double delta_call = exp(-q*T)*CDF_method.CDF(d1);
        result.first = V_call;
        result.second = delta_call;
        return result;
    };

     if (flag == "P" or flag == "p")
    {
        double d1 = (log(S / K) + (r + 0.5 * sigma * sigma - q) * T) / (sigma * sqrt(T));
        double d2 = d1 - sigma * sqrt(T);
        double V_put = -S * exp(-q * T) * CDF_method.CDF(-d1) + K * exp(-r * T) * CDF_method.CDF(-d2);
        double delta_put = exp(-q*T)*(CDF_method.CDF(d1)-1);
        result.first = V_put;
        result.second = delta_put;
        return result;
    };
};