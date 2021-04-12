/* 
* Copyright 2014-2020 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#ifndef POISSONFILEINPUTGROUPRANDOMRATE_H_
#define POISSONFILEINPUTGROUPRANDOMRATE_H_

#include <fstream>
#include <sstream>

#include <math.h>
#include <vector>

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "SpikingGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

namespace auryn {

/*! \brief Reads spikes from a ras file and emits them as SpikingGroup in a simulation.
 *
 * PoissonFileInputGroupRandomRate first reads the entire ras file into memory and emits the spikes then during the simulation
 * without file access. It supports looping over the input spikes by setting the loop argument with the constructor 
 * to true.
 *
 */
class PoissonFileInputGroupRandomRate : public SpikingGroup
{
private:
	// AurynTime next_event_time;
	// NeuronID next_event_spike;
	bool present_pattern;

	AurynTime reset_time;
	int actual_type;
	int neu;
  	AurynTime time_offset;
	
	// An array of vectors containing every spiking pattern
	std::vector<std::vector<SpikeEvent_type> > all_input_spikes;
	std::vector<SpikeEvent_type>::const_iterator spike_iter;

  	std::vector<AurynTime> input_events;
  	std::vector<AurynTime>::const_iterator event_iter;
	
	std::vector<int > type_events;
  	std::vector<int>::const_iterator type_iter;

	std::string fna;

  	AurynTime * clk;
	AurynDouble lambda;
  	AurynDouble lambda_p;
  	static boost::mt19937 gen; 
	boost::exponential_distribution<> * dist;
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > * die;

	unsigned int salt;


	void init(AurynDouble rate);

	AurynTime get_offset_clock();
  	void set_first_event();

protected:
  NeuronID x;

public:


	/*! \brief Default constructor */
	PoissonFileInputGroupRandomRate( NeuronID n, string filename_spikes , std::string filename_events, AurynDouble rate=5.,int nb_pattern =1);

	/*! \brief Constructor which does not load spikes.
	 *
	 * Use this constructor if you want to init the group without a filename containing spikes.
	 * Spikes can be added after initialization by calling load_spikes(filename) or by adding
	 * spikes one by one using add_spike(spiketime, neuron_id). */
	PoissonFileInputGroupRandomRate( NeuronID n, AurynDouble rate=5.0);

	virtual ~PoissonFileInputGroupRandomRate();
	virtual void evolve();

  void free();

	/*!\brief Aligned loop blocks to a temporal grid of this size 
	 *
	 * The grid is applied after the delay. It's mostly there to facilitate the
	 * synchronization of certain events or readouts.  If no grid is defined
	 * the last input spike time will define the end of the loop window (which
	 * often could be a fractional time). 
	 * */
	void set_loop_grid(AurynDouble grid_size);

	/*!\brief Load spikes from file
	 *
	 * */
	void load_spikes(std::string filename);
  void load_events(std::string filename);

        /*!\brief Adds a spike to the buffer manually and then temporally sorts the buffer
	 *
	 * \param spiketime the spike time 
	 * \param neuron_id the neuron you want to spike with neuron_id < size of the group
	 * */
	//void add_spike(double spiketime, NeuronID neuron_id=0);

	/*!\brief Sorts spikes by time
	 *
	 * Spikes need to be sorted befure a run otherwise the behavior of this SpikingGroup is undefined
	 * */
	void sort_spikes(int type_pattern);

	void set_rate(AurynDouble rate);
	/*! Standard getter for the firing rate variable. */
	AurynDouble get_rate();
	/*! Use this to seed the random number generator. */
	void seed(unsigned int s);
};

}

#endif /*POISSONFILEINPUTGROUPRANDOMRATE_H_*/
