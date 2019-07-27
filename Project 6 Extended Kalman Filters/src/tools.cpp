#include <iostream>
#include "tools.h"
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
	rmse<<0,0,0,0;
  // checking the validity of the estimations and ground_truth
  if(estimations.size() != ground_truth.size() || estimations.size() == 0){
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); i++){
    VectorXd residual = estimations[i] - ground_truth[i];
	//coefficient-wise multiplication
	residual = residual.array()*residual.array();
	rmse += residual;
  }

  //calculate the mean
  rmse = rmse/estimations.size();
  //calculate the squared root
  rmse = rmse.array().sqrt();
  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	//check division by zero
	if(fabs(c1) < 0.0001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}

VectorXd Tools::ConvertPolar2Cartesian(const VectorXd& polar_meas) {
	VectorXd cartesian_meas(4);
	const double rho = polar_meas(0);
  const double phi = polar_meas(1);
  const double rho_dot = polar_meas(2);

  const double px = rho * cos(phi);
  const double py = rho * sin(phi);
  const double vx = rho_dot * cos(phi);
  const double vy = rho_dot * sin(phi);

  cartesian_meas << px, py, vx, vy;
  return cartesian_meas;
}

VectorXd Tools::ConvertCartesian2Polar(const VectorXd& cartesian_meas) {
	VectorXd polar_meas(3);
	double px = cartesian_meas(0);
  double py = cartesian_meas(1);
  double vx = cartesian_meas(2);
  double vy = cartesian_meas(3);
	double temp1 = sqrt(px*px + py*py);
	double temp2 = px*vx + py*vy;
	if(fabs(temp1)<0.0001)
	{
		px += 0.001;
		py += 0.001;
		temp1 = sqrt(px*px + py*py);
	}
	double rho = temp1;
	double phi = atan2(py,px);
//   const double rho_dot = (rho>0.0001)?(px*vx + py*vy)/rho:0;
//   const double phi = (rho!=0)?atan2(py,px):0;
	double rho_dot = 0.0;
  if (fabs(rho) > 0.0001) {
    rho_dot = temp2/rho;
  }

  polar_meas << rho, phi, rho_dot;
  return polar_meas;
}