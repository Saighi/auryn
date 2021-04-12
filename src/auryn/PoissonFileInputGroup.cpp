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

#include "PoissonFileInputGroup.h"

using namespace auryn;

boost::mt19937 PoissonFileInputGroup::gen = boost::mt19937(); 

void PoissonFileInputGroup::init(AurynDouble  rate, AurynTime time_start_offset)
{
	auryn::sys->register_spiking_group(this);
	active = true;
	loop_grid_size = 1;
	reset_time = time_start_offset;

	if ( evolve_locally() ) {

	dist = new boost::exponential_distribution<>(1.0);;
	die  = new boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > ( gen, *dist );
	salt = sys->get_seed();
	seed(sys->get_seed());
	x = 0;
	set_rate( rate );

	}
}

PoissonFileInputGroup::PoissonFileInputGroup( NeuronID n, AurynDouble rate) : SpikingGroup(n, RANKLOCK ) 
{
	playinloop = false;
	time_delay = 0;
	time_start_offset =0;
	time_offset = 0;
	time_end_pat = 0;
	init(rate,time_start_offset);
}

PoissonFileInputGroup::PoissonFileInputGroup(NeuronID n, std::string filename, 
		bool loop, AurynFloat delay, AurynDouble rate, AurynFloat start_offset, AurynFloat end_pat) 
: SpikingGroup( n , RANKLOCK )
{
	playinloop = loop;
	if ( playinloop ) set_loop_grid(auryn_timestep);
	time_delay = (AurynTime) (delay/auryn_timestep);
	time_start_offset = (AurynTime) (start_offset/auryn_timestep);
	time_offset = 0;
	time_end_pat = (AurynTime) (end_pat/auryn_timestep);
	init(rate,time_start_offset);
	load_spikes(filename);
}

PoissonFileInputGroup::~PoissonFileInputGroup()
{
	if ( evolve_locally() ) {
	delete dist;
	delete die;
	}
}

bool time_compare (SpikeEvent_type a,SpikeEvent_type b) { return (a.time<b.time); }

void PoissonFileInputGroup::load_spikes(std::string filename)
{
	std::ifstream spkfile;
	input_spikes.clear();

	if ( evolve_locally() ) {
		spkfile.open(filename.c_str(),std::ifstream::in);
		if (!spkfile) {
			std::cerr << "Can't open input file " << filename << std::endl;
			std::exit(1);
		}
	}

	char buffer[255];
	while ( spkfile.getline(buffer, 256) ) {
		SpikeEvent_type event;
		std::stringstream line ( buffer ) ;
		double t_tmp;
		line >> t_tmp;
		event.time = round(t_tmp/auryn_timestep);
		line >> event.neuronID;
		if ( localrank(event.neuronID) ) {
			input_spikes.push_back(event);
		}
	}
	spkfile.close();

	sort_spikes();

	std::stringstream oss;
	oss << get_log_name() << ":: Finished loading " << input_spikes.size() 
		<< " spike events";
	logger->info(oss.str());

	spike_iter = input_spikes.begin();
}

void PoissonFileInputGroup::sort_spikes()
{
	std::sort (input_spikes.begin(), input_spikes.end(), time_compare);
}


AurynTime PoissonFileInputGroup::get_offset_clock() 
{
	return sys->get_clock() - time_offset;
}

AurynTime PoissonFileInputGroup::get_next_grid_point( AurynTime time ) 
{
	AurynTime result = time+time_delay;
	if ( result%loop_grid_size) { // align to temporal grid
		result = (result/loop_grid_size+1)*loop_grid_size;
	}
	return (result+auryn_timestep)-1;
}

void PoissonFileInputGroup::evolve()
{
	if (active && input_spikes.size()) {
		// when reset_time is reached reset the spike_iterator to The beginning and update time offset
		if ( sys->get_clock() == reset_time ) {
			spike_iter = input_spikes.begin(); 
			time_offset = sys->get_clock();
			// std::cout << "set to" << reset_time*auryn_timestep << " " << time_offset << std::endl;
		}

		while ( spike_iter != input_spikes.end() && (*spike_iter).time <= get_offset_clock() ) {
			spikes->push_back((*spike_iter).neuronID);
			++spike_iter;
			// std::cout << "spike " << sys->get_time() << std::endl;
		}

		// TODO Fix the bug which eats the first spike
		if ( spike_iter==input_spikes.end()) { // at last spike on file set new reset time

			if(reset_time < sys->get_clock() && playinloop){
				if (time_end_pat>0 && sys->get_clock() >= time_end_pat){
					reset_time = 0;
				}
				else{
					// schedule reset for next grid point after delay
					reset_time = get_next_grid_point(sys->get_clock());
					//std::cout << "oset rt" << reset_time << std::endl;
				}
				
			}else{
				while ( x < get_rank_size() ) {
					push_spike ( x );
					AurynDouble r = (*die)()/lambda;
					// we add 1.5: one to avoid two spikes per bin and 0.5 to 
					// compensate for rounding effects from casting
					x += (NeuronID)(r/auryn_timestep+1.5); 
					// beware one induces systematic error that becomes substantial at high rates, but keeps neuron from spiking twice per time-step
				}
				x -= get_rank_size();
			}
		}
	}
}

void PoissonFileInputGroup::set_loop_grid(AurynDouble grid_size)
{
	playinloop = true;
	if ( grid_size > 0.0 ) {
		loop_grid_size = 1.0/auryn_timestep*grid_size;
		if ( loop_grid_size == 0 ) loop_grid_size = 1;
	}
}

void PoissonFileInputGroup::add_spike(double spiketime, NeuronID neuron_id)
{
		SpikeEvent_type event;
		event.time = spiketime/auryn_timestep;
		event.neuronID = neuron_id;
		input_spikes.push_back(event);
		sort_spikes();
}

void PoissonFileInputGroup::set_rate(AurynDouble  rate)
{
	lambda = 1.0/(1.0/rate-auryn_timestep);
    if ( evolve_locally() ) {
		if ( rate > 0.0 ) {
		  AurynDouble r = (*die)()/lambda;
		  x = (NeuronID)(r/auryn_timestep+0.5); 
		} else {
			// if the rate is zero this triggers one spike at the end of time/groupsize
			// this is the easiest way to take care of the zero rate case, which should 
			// be avoided in any case.
			x = std::numeric_limits<NeuronID>::max(); 
		}
    }
}

AurynDouble  PoissonFileInputGroup::get_rate()
{
	return lambda;
}

void PoissonFileInputGroup::seed(unsigned int s)
{
	std::stringstream oss;
	oss << "PoissonGroup:: Seeding with " << s
		<< " and " << salt << " salt";
	auryn::logger->msg(oss.str(),NOTIFICATION);

	gen.seed( s + salt );  
}

