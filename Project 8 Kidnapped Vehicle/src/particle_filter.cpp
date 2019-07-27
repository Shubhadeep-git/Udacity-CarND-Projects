/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;
	
	//normal distributions for sensor noise
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);
	
	for(int i=0; i<num_particles; i++)
	{
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(p.weight);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0,std_pos[1]);
	normal_distribution<double> dist_theta(0,std_pos[2]);
	
	for(int i=0; i<num_particles; i++)
	{
		Particle p = particles[i];
		
		if(fabs(yaw_rate)<0.0001)
		{
			particles[i].x += velocity * delta_t * cos(p.theta);
			particles[i].y += velocity * delta_t * sin(p.theta);
		}
		else 
		{
			particles[i].x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
			particles[i].y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		
		// adding noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0; i<observations.size(); i++)
	{
		LandmarkObs obs = observations[i];
		double min_dist = numeric_limits<double>::max();
		for(int j=0; j<predicted.size(); j++)
		{
			LandmarkObs pred = predicted[j];
			double calc_dist = dist(obs.x,obs.y,pred.x,pred.y);
			if(min_dist>calc_dist)
			{
				min_dist = calc_dist;
				observations[i].id = j;
			}				
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	for(int i=0; i<num_particles; i++)
	{
		Particle p = particles[i];
		vector<LandmarkObs> predicted;
		for(int j=0; j<map_landmarks.landmark_list.size(); j++)
		{
			LandmarkObs pred;
			pred.id = map_landmarks.landmark_list[j].id_i;
			pred.x = map_landmarks.landmark_list[j].x_f;
			pred.y = map_landmarks.landmark_list[j].y_f;
			if(fabs(pred.x-p.x) <= sensor_range && fabs(pred.y-p.y) <= sensor_range) 
			{
				predicted.push_back(pred);
			}
		}
		vector<LandmarkObs> transformed_obs;
		for(int j = 0; j<observations.size(); j++)
		{
			LandmarkObs obs = observations[j];
			LandmarkObs trans;
			trans.id = obs.id;
			trans.x = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y;
			trans.y = p.y + sin(p.theta)*obs.x + cos(p.theta)*obs.y;
			transformed_obs.push_back(trans);
		}
		
		dataAssociation(predicted,transformed_obs);
		
		double probab = 1.0; 
		for(int j=0; j<observations.size(); j++)
		{
			LandmarkObs obs = transformed_obs[j];
			LandmarkObs pred = predicted[obs.id];
			
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];
			double gauss_norm = (1/(2*M_PI*sig_x*sig_y));
			double exponent = (pow(obs.x-pred.x,2)/(2*pow(sig_x,2)) + (pow(obs.y-pred.y,2)/(2*pow(sig_y, 2))));
			probab *= gauss_norm * exp(-exponent);
		}
		particles[i].weight = probab;
		weights[i] = particles[i].weight;	
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resampled_particles;
	vector<double> weights_;
	for(Particle p:particles)
		weights_.push_back(p.weight);
	double w_max = *max_element(weights_.begin(),weights_.end());
	uniform_int_distribution<int> dist_index(0, num_particles-1);
	int index = dist_index(gen);
	uniform_real_distribution<double> dist_beta(0.0, w_max);
	double beta = 0.0;
	for(int i=0; i<num_particles; i++)
	{
		beta += dist_beta(gen)*2.0;
		while(beta>weights_[index])
		{
			beta -= weights_[index];
			index = (index+1)%num_particles;
		}
		resampled_particles.push_back(particles[index]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
