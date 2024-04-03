#ifndef Option_h
#define Option_h
#include<iostream>
#include <string>
using namespace std;

class Option {
private:
    double K, S, r, option_price; string t_0, t_N, flag;

    void init(double Strike_price, double underlying_asset, double Option_price, double risk_free_rate, string flag_PC, string t0, string tN);

public:
    // Default Constructor
    Option();

    // Constructor with parameters
    Option(double Strike_price, double underlying_asset, double Option_price, double risk_free_rate, string flag_PC, string t0, string tN);

    // Destructor
    ~Option();

    // get() method for each parameter
    double get_K() const;
    double get_S() const;
    double get_r() const;
    double get_option_price() const;
    string get_t_0() const;
    string get_t_N() const;
    string get_flag() const;

    int countBusinessDays(struct tm start_tm, struct tm end_tm);
    double get_T(string start_date, string end_date);

    pair<double, double> BSM_Pricer(double volatility);
};


#endif //Option_h