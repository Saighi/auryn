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

#include "PoissonFileInputGroupRandomRate.h"
#include "auryn_definitions.h"
#include <fstream>
#include <string>

using namespace auryn;

boost::mt19937 PoissonFileInputGroupRandomRate::gen = boost::mt19937(); 

void PoissonFileInputGroupRandomRate::init(AurynDouble  rate)
{
	auryn::sys->register_spiking_group(this);
	active = true;

	if ( evolve_locally() ) {

	dist = new boost::exponential_distribution<>(1.0);
	die  = new boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > ( gen, *dist );
	salt = sys->get_seed();
	seed(sys->get_seed());
	x = 0;
	set_rate( rate );
  }
}

PoissonFileInputGroupRandomRate::PoissonFileInputGroupRandomRate( NeuronID n, AurynDouble rate) : SpikingGroup(n, RANKLOCK ) 
{
	present_pattern = false;
	init(rate);
}

PoissonFileInputGroupRandomRate::PoissonFileInputGroupRandomRate(
                                                         NeuronID n, std::string filename_spikes ,std::string filename_events, AurynDouble rate, int nb_pattern)

    : SpikingGroup(n, RANKLOCK) {

  all_input_spikes.resize(nb_pattern);
  present_pattern = true;
  init(rate);
  load_events(filename_events);
  set_first_event();
  actual_type = 0;
  load_spikes(filename_spikes);
  fna = filename_spikes;
}

void PoissonFileInputGroupRandomRate::set_first_event(){
	reset_time = (*event_iter);
	++event_iter;
} 

bool time_compare (SpikeEvent_type a,SpikeEvent_type b) { return (a.time<b.time); }

void PoissonFileInputGroupRandomRate::load_spikes(std::string filename)
{
	std::cout << "spike_loading" << std::endl;
	char strbuf [255]; // Buffer to store filename depending of type of pattern
	for (int v = 0; v<all_input_spikes.size(); ++v){
	
		std::ifstream spkfile;
		all_input_spikes[v].clear();

		if ( evolve_locally() ) {
        	sprintf(strbuf, "%s_%i",filename.c_str(),v);
			spkfile.open(string(strbuf),std::ifstream::in);
			if (!spkfile) {
				std::cerr << "Can't open input file " << string(strbuf) << std::endl;
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
				all_input_spikes[v].push_back(event);
			}
		}
		
		spkfile.close();

		sort_spikes(v);

		std::stringstream oss;
		oss << get_log_name() << ":: Finished loading " << all_input_spikes[v].size() 
			<< " spike events";
		logger->info(oss.str());

		
	}

	spike_iter = all_input_spikes[actual_type].end();
}

void PoissonFileInputGroupRandomRate::load_events(std::string filename) {
	
  std::ifstream eventfile;
  input_events.clear();
  type_events.clear();

  //if (evolve_locally()) {
  eventfile.open(filename.c_str(), std::ifstream::in);
  if (!eventfile) {
	std::cerr << "Can't open input file " << filename << std::endl;
	std::exit(1);
  }
  //}

  AurynTime time_event;
  int type_event;

  while (eventfile>>time_event>>type_event) {
    input_events.push_back(time_event);
	type_events.push_back(type_event);
  }
  eventfile.close();

  event_iter = input_events.begin();
  type_iter = type_events.begin();
}

void PoissonFileInputGroupRandomRate::sort_spikes(int type_pattern)
{
	std::sort (all_input_spikes[type_pattern].begin(), all_input_spikes[type_pattern].end(), time_compare);
}


AurynTime PoissonFileInputGroupRandomRate::get_offset_clock() 
{
	return sys->get_clock() - time_offset;
}

void PoissonFileInputGroupRandomRate::evolve()
{

	if (sys->get_clock() == 100000){
		load_spikes(fna);
	}

	if (active && present_pattern ) {
		// when reset_time is reached reset the spike_iterator to The beginning and update time offset
		if ( sys->get_clock() == reset_time ) {
                  actual_type = (*type_iter);
                  ++type_iter;
                  spike_iter = all_input_spikes[actual_type].begin();
                  time_offset = sys->get_clock();
                  // std::cout << "new_pattern" << sys->get_clock() << " " <<
                  // time_offset << std::endl;
		}

		while ( spike_iter != all_input_spikes[actual_type].end() && (*spike_iter).time <= get_offset_clock() ) {
			spikes->push_back((*spike_iter).neuronID);
			++spike_iter;
			// std::cout << "spike " << sys->get_time() << std::endl;
		}

		// TODO Fix the bug which eats the first spike
		if ( spike_iter==all_input_spikes[actual_type].end()) { // at last spike on file set new reset time
			if (reset_time < sys->get_clock()) {
				
				if (event_iter != input_events.end()) {
				reset_time = (*event_iter);
				++event_iter;
				} else {
          reset_time = std::numeric_limits<AurynTime>::max();
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
	}else if(active){
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



// void PoissonFileInputGroupRandomRate::add_spike(double spiketime, NeuronID neuron_id)
// {
// 		SpikeEvent_type event;
// 		event.time = spiketime/auryn_timestep;
// 		event.neuronID = neuron_id;
// 		input_spikes.push_back(event);
// 		sort_spikes();
// }

void PoissonFileInputGroupRandomRate::set_rate(AurynDouble  rate)
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

AurynDouble  PoissonFileInputGroupRandomRate::get_rate()
{
	return lambda;
}


void PoissonFileInputGroupRandomRate::seed(unsigned int s)
{
	std::stringstream oss;
	oss << "PoissonGroup:: Seeding with " << s
		<< " and " << salt << " salt";
	auryn::logger->msg(oss.str(),NOTIFICATION);

	gen.seed( s + salt );  
}

PoissonFileInputGroupRandomRate::~PoissonFileInputGroupRandomRate()
{
	if ( evolve_locally() ) {
    delete dist;
    delete die;
	}
}
