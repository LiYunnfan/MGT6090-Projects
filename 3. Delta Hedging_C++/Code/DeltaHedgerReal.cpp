
#include "DeltaHedgerReal.h"
#include "StdNormalCDF.h"
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <random>
#include <iostream>
#include <utility>

using namespace std;

DeltaHedgerReal::DeltaHedgerReal() {}

DeltaHedgerReal::DeltaHedgerReal(double strick_price, string t0, string tN, string T_date) {
    t_N = tN;
    t_0 = t0;
    T = T_date;
    K = strick_price;
    read_option_csv();
    calculate_implied_volatility();
    calculate_Delta();
    calculate_PNL_Hedge();
    calculate_PNL();
    write_csv();
}

DeltaHedgerReal::~DeltaHedgerReal(){};

// Find data corresponding a specific date
double DeltaHedgerReal::find_data(vector<double> rate_vector, vector<string> date_vector, string date)
{
    for (int i = 0; i < date_vector.size(); ++i) {
        if (date == date_vector[i]){
            return rate_vector[i];
        }
    }
}

// Read Interest File
pair<vector<string>, vector<double> > DeltaHedgerReal::read_interest_csv() {
    vector<string> date_vector;
    vector<double> rate_vector;
    string line, str1, str2;
    pair<vector<string>, vector<double> > date_interest;

    ifstream infile("./interest.csv");
    if (infile.fail()) {cout << "Opening File failed" << endl;}

    getline(infile, line);
    while (!infile.fail()) {
        getline(infile, line);
        stringstream linestream(line);
        if (!linestream.str().empty()) {
            getline(linestream, str1, ',');
            date_vector.push_back(str1);
            getline(linestream, str2, ',');
            rate_vector.push_back(stod(str2) / 100);
        }
    }
    infile.close();
    date_interest.first = date_vector;
    date_interest.second = rate_vector;
    return date_interest;
}

// Read Stock Price File
pair<vector<string>, vector<double> > DeltaHedgerReal::read_stock_csv() {
    vector<string> date_vector;
    vector<double> stock_price_vector;
    string line, str1, str2;
    pair<vector<string>, vector<double> > date_price;

    ifstream infile("./sec_GOOG.csv");
    if (infile.fail()) {cout << "Opening File failed" << endl;}

    getline(infile, line);
    while (!infile.fail()) {
        getline(infile, line);
        stringstream linestream(line);
        if (!linestream.str().empty()) {
            getline(linestream, str1, ',');
            date_vector.push_back(str1);
            getline(linestream, str2, ',');
            stock_price_vector.push_back(stod(str2));
        }
    }
    date_price.first = date_vector;
    date_price.second = stock_price_vector;
    return date_price;
}

// Read Option Price File and Save Option Class In a Vector
vector<Option> DeltaHedgerReal::read_option_csv() {
    vector<Option> option_vector;
    vector<string> date_rate_vector, date_price_vector;
    vector<double> rate_vector, stock_price_vector;
    string line, start_date, expired_date, flag, strike_price, Bid, Ask;
    double option_price, sopt, risk_free_rate;
    pair<vector<string>, vector<double> > date_interest,date_price;

    date_interest = read_interest_csv();
    date_rate_vector = date_interest.first;
    rate_vector = date_interest.second;
    // cout << date_rate_vector[1] << endl;

    date_price = read_stock_csv();
    date_price_vector = date_price.first;
    stock_price_vector = date_price.second;
    // cout << date_price_vector[1] << endl;

    ifstream infile("./op_GOOG.csv");
    if (infile.fail()) {cout << "Opening File failed" << endl;}

    getline(infile, line);
    while (!infile.fail()) {
        getline(infile, line);
        stringstream linestream(line);
        if (!linestream.str().empty()) {
            getline(linestream, start_date, ',');
            getline(linestream, expired_date, ',');
            getline(linestream, flag, ',');
            getline(linestream, strike_price, ',');
            getline(linestream, Bid, ',');
            getline(linestream, Ask, ',');

            if (expired_date == T && stod(strike_price) == K && flag == "C") {
                option_price = (stod(Bid) + stod(Ask)) / 2;
                sopt = find_data(stock_price_vector, date_price_vector, start_date);
                risk_free_rate = find_data(rate_vector, date_rate_vector, start_date);
                auto *flag_PC = reinterpret_cast<string *>(&flag[0]);
                Option option(stod(strike_price), sopt, option_price, risk_free_rate, *flag_PC, start_date, expired_date);
                option_vector.push_back(option);
            }
        }
    }

// Get Option in a specific date range
    int index_t_0 = 0;
    int index_t_N = 0;
    for (int i = 0; i < option_vector.size(); i++) {
        if (option_vector[i].get_t_0() == t_0) {
            index_t_0 = i;
        }
        if (option_vector[i].get_t_0() == t_N) {
            index_t_N = i + 1;
        }
    }
    vector<Option> Option_0_N(&option_vector[index_t_0], &option_vector[index_t_N]);
    Option_0_N_copy = Option_0_N;

    return Option_0_N;
}

// Find volatility to make the differences between BSM option price and market option price mininum.
vector<double> DeltaHedgerReal::calculate_implied_volatility() {

    vector<double> implied_vol_vector, volatility_vector;
    double min_value, implied_volatility, option_price;
    
    for (double s = 0.01; s <= 0.99; s = s + 0.001) {
        volatility_vector.push_back(s);
    }

    for (auto option_element : Option_0_N_copy) {
        min_value = 10000000; 
        for (double i : volatility_vector) {
            option_price = option_element.BSM_Pricer(i).first;
            if (abs(option_element.get_option_price() - option_price) < min_value) {
                implied_volatility = i;
                min_value = abs(option_element.get_option_price() - option_price);
            }
        }
        implied_vol_vector.push_back(implied_volatility);
    }
    implied_volitility_copy = implied_vol_vector;

    return implied_vol_vector;
}

// Find volatility to make the differences between BSM option price and market option price mininum.
vector<double> DeltaHedgerReal::calculate_Delta() {

    vector<double> delta_vector;
    double delta;

    for (int i = 0; i < Option_0_N_copy.size(); i++) {

        delta = Option_0_N_copy[i].BSM_Pricer(implied_volitility_copy[i]).second;
        delta_vector.push_back(delta);
    }
    delta_vector_copy = delta_vector;

    return delta_vector;
}

// Find the PNL Hedge(the same as Hedge Error)
vector<double> DeltaHedgerReal::calculate_PNL_Hedge() {

    vector<double> B_vector, HE_vector;
    double B_i, HE_i;

    for (int i = 0; i < Option_0_N_copy.size(); i++) {
        if (i == 0) {
            B_i = Option_0_N_copy[i].get_option_price() - delta_vector_copy[i] * Option_0_N_copy[i].get_S();
        }
        else {
            B_i = (delta_vector_copy[i - 1] - delta_vector_copy[i]) * Option_0_N_copy[i].get_S() +
                    B_vector[i - 1] * exp(Option_0_N_copy[i].get_r() / 252);
        }
        B_vector.push_back(B_i);
    }

    for (int i = 0; i < Option_0_N_copy.size(); i++) {
        if (i == 0) {
            HE_i = 0;
        }
        else {
            HE_i = delta_vector_copy[i - 1] * Option_0_N_copy[i].get_S() +
                 B_vector[i - 1] * exp(Option_0_N_copy[i].get_r() / 252) - Option_0_N_copy[i].get_option_price();
        }
        HE_vector.push_back(HE_i);
    }

    PNL_with_hedge = HE_vector;

    return HE_vector;
}

// Find the PNL
vector<double> DeltaHedgerReal::calculate_PNL() {

    vector<double> PNL_vector;
    double PNL_value;

    for (auto & option_i : Option_0_N_copy) {
        PNL_value = Option_0_N_copy[0].get_option_price() - option_i.get_option_price();
        PNL_vector.push_back(PNL_value);
    }
    PNL_copy = PNL_vector;

    return PNL_vector;
}

// Write them into a csv
void DeltaHedgerReal::write_csv() {
    ofstream output;
    output.open("./output.csv");
    output << "Date" << "," << "Stock Price" << "," << "Option Price" << ","
            << "Implied Volatility" << "," << "HE"<< ","<< "Delta" << "," << "PNL" << ","
            << "PNL Hedged" << endl;

    for (int i = 0; i < Option_0_N_copy.size(); i++) {

        output << Option_0_N_copy[i].get_t_0() << ","
        << Option_0_N_copy[i].get_S() << ","
        << Option_0_N_copy[i].get_option_price() << ","
        << implied_volitility_copy[i] << "," 
        << PNL_with_hedge[i] << "," 
        << delta_vector_copy[i]<< "," 
        << PNL_copy[i] << ","
        << PNL_with_hedge[i] 
        << endl;
    }
    output.close();
}