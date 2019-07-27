#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 10;
double dt = 0.1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

double v_desired = 100;
double cte_desired = 0;
double epsi_desired = 0;

size_t ini_x = 0;
size_t ini_y = ini_x + N;
size_t ini_psi = ini_y + N;
size_t ini_v = ini_psi + N;
size_t ini_cte = ini_v + N;
size_t ini_epsi = ini_cte + N;
size_t ini_delta = ini_epsi + N;
size_t ini_a = ini_delta + N -1;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.
	fg[0] = 0;
	
	double W_cte = 2000;
	double W_epsi = 2000;
	double W_v = 1;
	double W_delta = 70;
	double W_a = 1;
	double W_ddelta = 500;
	double W_da = 1;
	
	for(unsigned int i=0; i<N; i++)
	{
		fg[0] += W_v*CppAD::pow(vars[ini_v+i]-v_desired,2);
		fg[0] += W_cte*CppAD::pow(vars[ini_cte+i]-cte_desired,2);
		fg[0] += W_epsi*CppAD::pow(vars[ini_epsi+i]-epsi_desired,2);
	}
	
	for(unsigned int i=0; i<N-1; i++)
	{
		fg[0] += W_delta*CppAD::pow(vars[ini_delta+i],2);
		fg[0] += W_a*CppAD::pow(vars[ini_a+i],2);
	}
	
	for(unsigned int i=0; i<N-2; i++)
	{
		fg[0] += W_ddelta*CppAD::pow(vars[ini_delta+i+1]-vars[ini_delta+i],2);
		fg[0] += W_da*CppAD::pow(vars[ini_a+i+1]-vars[ini_a+i],2);
	}
	
	fg[1+ini_x] = vars[ini_x];
	fg[1+ini_y] = vars[ini_y];
	fg[1+ini_psi] = vars[ini_psi];
	fg[1+ini_v] = vars[ini_v];
	fg[1+ini_cte] = vars[ini_cte];
	fg[1+ini_epsi] = vars[ini_epsi];
	
	for(unsigned int j=0; j<N-1; j++)
	{
		AD<double> x_0 = vars[ini_x+j];
		AD<double> y_0 = vars[ini_y+j];
		AD<double> psi_0 = vars[ini_psi+j];
		AD<double> v_0 = vars[ini_v+j];
		AD<double> cte_0 = vars[ini_cte+j];
		AD<double> epsi_0 = vars[ini_epsi+j];
		AD<double> delta_0 = vars[ini_delta+j];
		AD<double> a_0 = vars[ini_a+j];
		
		AD<double> x_1 = vars[ini_x+j+1];
		AD<double> y_1 = vars[ini_y+j+1];
		AD<double> psi_1 = vars[ini_psi+j+1];
		AD<double> v_1 = vars[ini_v+j+1];
		AD<double> cte_1 = vars[ini_cte+j+1];
		AD<double> epsi_1 = vars[ini_epsi+j+1];
		
		AD<double> f_0 = coeffs[3]*pow(x_0,3) + coeffs[2]*pow(x_0,2) + coeffs[1]*x_0 + coeffs[0];
		AD<double> psi_d0 = CppAD::atan(3*coeffs[3]*pow(x_0,2) + 2*coeffs[2]*x_0 + coeffs[1]);
		
		fg[2+ini_x+j] = x_1 - (x_0 + v_0*CppAD::cos(psi_0)*dt);
		fg[2+ini_y+j] = y_1 - (y_0 + v_0*CppAD::sin(psi_0)*dt);
		fg[2+ini_psi+j] = psi_1 - (psi_0 - (v_0/Lf)*delta_0*dt);
		fg[2+ini_v+j] = v_1 - (v_0 + a_0*dt);
		fg[2+ini_cte+j] = cte_1 - ((f_0 - y_0) + v_0*CppAD::sin(epsi_0)*dt);
		fg[2+ini_epsi+j] = epsi_1 - ((psi_0 - psi_d0) - (v_0/Lf)*delta_0*dt);
	}
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  //size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9
  
  double px = state[0];
  double py = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];
  
  size_t n_vars = state.size() * N + 2 * (N-1);
  // TODO: Set the number of constraints
  size_t n_constraints = state.size() * N;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (unsigned int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.
  for(unsigned int i=0; i<ini_delta; i++)
  {
	  vars_lowerbound[i] = -1.0e19;
	  vars_upperbound[i] = 1.0e19;
  }
  for(unsigned int i=ini_delta; i<ini_a; i++)
  {
	  vars_lowerbound[i] = -0.436332*Lf;
	  vars_upperbound[i] = 0.436332*Lf;
  }
  for(unsigned int i=ini_a; i<n_vars; i++)
  {
	  vars_lowerbound[i] = -1.0;
	  vars_upperbound[i] = 1.0;
  }
  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (unsigned int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[ini_x] = px;
  constraints_upperbound[ini_x] = px;

  constraints_lowerbound[ini_y] = py;
  constraints_upperbound[ini_y] = py;

  constraints_lowerbound[ini_psi] = psi;
  constraints_upperbound[ini_psi] = psi;

  constraints_lowerbound[ini_v] = v;
  constraints_upperbound[ini_v] = v;

  constraints_lowerbound[ini_cte] = cte;
  constraints_upperbound[ini_cte] = cte;

  constraints_lowerbound[ini_epsi] = epsi;
  constraints_upperbound[ini_epsi] = epsi;
  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  vector<double> output;
  output.push_back(solution.x[ini_delta]);
  output.push_back(solution.x[ini_a]);
  for(unsigned int j=0; j<N-1; j++)
  {
	  output.push_back(solution.x[ini_x+j+1]);
	  output.push_back(solution.x[ini_y+j+1]);
  }
  return output;
}
