#include "StdNormalCDF.h"

double StdNormalCDF::standardNormalCDF(double x)
    {
        double a1 = 0.319381530;
        double a2 = -0.356563782;
        double a3 = 1.781477937;
        double a4 = -1.821255978;
        double a5 = 1.330274429;
        double z = 1/(1+0.2316419*abs(x));
        double Rz = a1*z+a2*pow(z,2)+a3*pow(z,3)+a4*pow(z,4)+a5*pow(z,5);
        if (x >= 0)
        {
            return 1-1/sqrt(2*M_PI)*exp(-pow(x,2)/2)*Rz;
        }
        else
        {
            return 1/sqrt(2*M_PI)*exp(-pow(x,2)/2)*Rz;
        }
    };

double StdNormalCDF::CDF(double x) {
        return StdNormalCDF::standardNormalCDF(x);
    };