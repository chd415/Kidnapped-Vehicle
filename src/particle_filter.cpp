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

using namespace std;
//static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 200;
    default_random_engine gen;
  
  	normal_distribution<double> dist_x_ini(x, std[0]);
 	  normal_distribution<double> dist_y_ini(y, std[1]);
  	normal_distribution<double> dist_theta_ini(theta, std[2]);
  
  	for (int i = 0; i < num_particles; i++){
      Particle par;
      par.id = i;
      par.x = dist_x_ini(gen);
      par.y = dist_y_ini(gen);
      par.theta = dist_theta_ini(gen);
      par.weight = 1.0;
      particles.push_back(par);
      weights.push_back(par.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  	default_random_engine gen;
  
  	for (int i = 0; i < num_particles; i++){
  		double particle_x = particles[i].x;
  		double particle_y = particles[i].y;
	  	double particle_theta = particles[i].theta;

	  	double pred_x;
	  	double pred_y;
	  	double pred_theta;

    if (fabs(yaw_rate) < 0.00001) {  
    	pred_x = particle_x + velocity * delta_t * cos(particle_theta);
    	pred_y = particle_y + velocity * delta_t * sin(particle_theta);
    	pred_theta = particle_theta;
    } 
    else {
    	pred_x = particle_x + (velocity / yaw_rate) * (sin(particle_theta + (yaw_rate*delta_t)) - sin(particle_theta));
    	pred_y = particle_y + (velocity / yaw_rate) * (cos(particle_theta) - cos(particle_theta + (yaw_rate * delta_t)));
    	pred_theta = particle_theta + (yaw_rate * delta_t);
      
    }

    normal_distribution<double> dist_x_pre(pred_x, std_pos[0]);
 	  normal_distribution<double> dist_y_pre(pred_y, std_pos[1]);
  	normal_distribution<double> dist_theta_pre(pred_theta, std_pos[2]);

    particles[i].x = dist_x_pre(gen);
    particles[i].y = dist_y_pre(gen);
    particles[i].theta = dist_theta_pre(gen);
 
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i=0; i < observations.size(); i++){
      	double min_dist = sensor_range * sqrt(2);
      	int map_id = -1;
      	double obs_x = observations[i].x;
      	double obs_y = observations[i].y;
      
      	for (unsigned int j=0; j<predicted.size();j++){
      		double pred_x = predicted[j].x;
      		double pred_y = predicted[j].y;
          int pred_id = predicted[j].id;
          double distanceop = dist(obs_x, obs_y, pred_x, pred_y);
          if (distanceop < min_dist){
             	min_dist = distanceop;
              map_id = pred_id;
            }
        }
      	observations[i].id = map_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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
  
  	double weight_normalizer = 0.0;
  
  	for (int i=0; i<num_particles; i++) {
     	double p_x = particles[i].x;
      double p_y = particles[i].y;
      double p_theta = particles[i].theta;
      
      vector<LandmarkObs> transformed_pos;
      
        for (unsigned int j = 0; j < observations.size(); j++) {
      		LandmarkObs transformed_position;
      		transformed_position.id = j;
      		transformed_position.x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      		transformed_position.y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      		transformed_pos.push_back(transformed_position);
      	}

       vector<LandmarkObs> predictions;
      	for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++){

          	Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
          
          	if ( (fabs((current_landmark.x_f - p_x)) <= sensor_range) && (fabs(current_landmark.y_f - p_y) <= sensor_range)) {
             	predictions.push_back(LandmarkObs{ current_landmark.id_i, current_landmark.x_f, current_landmark.y_f}); 
            }
        }
      
      dataAssociation(predictions, transformed_pos, sensor_range);
      	
      	particles[i].weight = 1.0;
      
      double s_x = std_landmark[0];
    	double s_y = std_landmark[1];
    	double s_x_2 = pow(s_x, 2);
    	double s_y_2 = pow(s_y, 2);
    	double normalizer = (1.0/(2.0 * M_PI * s_x * s_y));

      	for (unsigned int l = 0; l < transformed_pos.size(); l++) {
      
      		double trans_x, trans_y, trans_id, pr_x, pr_y, pr_id;
      		trans_x = transformed_pos[l].x;
      		trans_y = transformed_pos[l].y;
          trans_id = transformed_pos[l].id;
          double multi_prob = 1.0;

	       	for (unsigned int k = 0; k < predictions.size(); k++) {
	       		pr_id = predictions[k].id;
    		    if (pr_id == trans_id) {
                  
          			pr_x = predictions[k].x;
          			pr_y = predictions[k].y;
             
                  	multi_prob = normalizer * exp(-1.0 * ((pow((trans_x - pr_x), 2)/(2.0 * s_x_2)) + (pow((trans_y - pr_y), 2)/(2.0 * s_y_2))));
                  	particles[i].weight *= multi_prob;
        		}
      		}
			/**
          	double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(pr_x-trans_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-trans_y,2)/(2*pow(s_y, 2))) ) );
          	particles[i].weight *= obs_w;
            */
      	}
      	weight_normalizer += particles[i].weight;
    }

  	for (unsigned int q = 0; q < particles.size(); q++) {
    	particles[q].weight /= weight_normalizer;
    	weights[q] = particles[q].weight;
  	}

  
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;
  
  	default_random_engine gen;
  
  	/*
  	vector<double> weights;
  	for (int i=0; i<=num_particles; ++i){
     	weights.push_back(particles[i].weight); 
    }
   */
  
  	uniform_int_distribution<int> uniintdist(0, num_particles-1);
  	auto index = uniintdist(gen);
  	
  	double beta = 0.0;
  	double mw_2 = 2.0 * *max_element(weights.begin(), weights.end());
  	
  	for (unsigned int i=0; i<particles.size(); i++){
      	uniform_real_distribution<double> unirealdist(0.0, mw_2);
     	  beta += unirealdist(gen);
      	while (beta > weights[index]){
         	beta -= weights[index];
          index = (index + 1) % num_particles;
        }
      	new_particles.push_back(particles[index]);
    }
  	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

  	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();
  
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
