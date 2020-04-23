#!/usr/bin/env python

from collections import namedtuple
import glob
import os.path
import sys

import numpy as np
import scipy.io
import yaml

from slopey2 import analyze
from slopey2.analyze import TraceData, SamplerState, SlopeyDurationDistn, TraceLatentVars
from slopey2.trace_mcmc import TraceLatentVars


class SkipTrace: pass

def load_trace_file(fname):
  # if there's a corresponding .params.yml file, use it for start/end/discard
  basename, _ = os.path.splitext(fname)
  trace_config_fname = os.path.join(basename, '.params.yml')
  if os.path.isfile(trace_config_fname):
    trace_config = yaml.safe_load(trace_config_fname)
    discard = trace_config['discard']
    start, end = trace_config['start'], trace_config['end']
  else:
    discard = False
    start = 0
    end = None

  if discard:
    return skip_trace
  else:
    datadict = scipy.io.loadmat(fname)
    get = lambda name: np.squeeze(datadict[name])

    # extract red and green channels
    red_frames = get('unsmoothedRedI')
    green_frames = get('unsmoothedGrI')
    frames = np.hstack([red_frames[:, None], green_frames[:, None]])  # (N, 2)
    frames = frames[start:end, :]

    # get translocation frame guesses from HMM fit for initialization
    durations = get('model_durations').ravel()
    translocation_guesses = np.cumsum(durations[:-1])

    return TraceData(frames, translocation_guesses)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('{} data/experiment_dir'.format(sys.argv[0]), file=sys.stderr)
    sys.exit(1)
  _, datadir = sys.argv

  config_file = os.path.join(datadir, 'params.yml')
  with open(config_file) as f:
    config = yaml.safe_load(f)

  trace_files = glob.glob(os.path.join(datadir, '*.mat'))
  traces = {}
  for fname in trace_files:
    trace_data = load_trace_file(fname)
    if type(trace_data) is SkipTrace:
      continue
    elif type(trace_data) is TraceData:
      name, _ = os.path.splitext(os.path.basename(fname))
      traces[name] = trace_data
    else:
      raise TypeError(trace_data)

  samples = analyze.run_mcmc(config, traces)
  
  # Save samples: Steph contemplations:
  # Current plan is to save results in more or less the same format as my Matlab
  # plotting code expects, to limit the rewriting of that code that will be needed.
  # WILL NEED TO REWRITE DEPENDENCE ON MAKE and ability to re-analyze single traces

  # Samples is a list of SamplerStates, one for each iteration of the MCMC,
  # with the first element the initialization, i.e.
  # samples = [SamplerState_Init, SamplerState_Iter1, SamplerStateIter2 ...]
  # Each SamplerState is a named tuple composed of a named tuple, SlopeyDurationDistn,
  # which contains the a and b parameters that parameterize the gamma on slopeys, and
  # a dict of trace_name : TraceLatentVars named tuple key-value pairs.
  # TraceLatentVars contains named tuple elemnts 'times', 'red_vals', 'time_offset', 
  # 'green_channel_transform_slope', 'green_channel_transform_intercept', 'camera_noise_sigma'
  # So:
  # samples[0].SlopyDurationDistn.a = a_init
  # samples[0].SlopyDurationDistn.b = b_init
  # samples[0].trace_latents["trace_name"].TraceLatentVars.times = initialization of slopey start and end points (translocation_guesses)

  # Will need to get trace_names from traces.keys(), and data from traces.values()
  # params should be a combination of config and any trace-specific parameters loaded via load_trace_file.
  # I think for now I'm not going to include trace-specific parameters, but opened an issue about this.

 
  trace_names = samples[0].trace_latents.keys()

  trace_vals = []
  for t in trace_names:
  	one_trace_vals = {}
  	one_trace_vals["params"]=config
  	one_trace_vals["data"]=traces[t].trace_data.frames

    times_samples = np.empty(shape=(len(samples),len(samples[0].trace_latents[t].TraceLatentVars.times)))
    vals_samples = np.empty(shape=(len(samples),len(samples[0].trace_latents[t].TraceLatentVars.red_vals))) # called red_vals in slopey_hierarchical
    u_samples = np.empty(shape=(len(samples),1)) # called time_offset in slopey_hierarchical
    ch2_samples = np.empty(shape=(len(samples),2)) # called green_channel_transform_slope, green_channel_transform_intercept
    sigma_samples = np.empty(shape=(len(samples),1)) # called camera_noise_sigma in slopey_hierarchical
    ab_samples = np.empty(shape=(len(samples),2))

    for s in range(0:len(samples)):
      times_samples[s,:] = samples[s].trace_latents[t].TraceLatentVars.times
      vals_samples[s,:] = samples[s].trace_latents[t].TraceLatentVars.red_vals
      u_samples[s] = samples[s].trace_latents[t].TraceLatentVars.time_offset
      # STEPH TO DO does matlab expect slope, intercept for ch2_samples?
      ch2_samples[s,:] = np.array([samples[s].trace_latents[t].TraceLatentVars.green_channel_transform_slope,
                                   samples[s].trace_latents[t].TraceLatentVars.green_channel_transform_intercept
                           ]
                         )
      sigma_samples[s] = samples[s].trace_latents[t].TraceLatentVars.camera_noise_sigma
      ab_samples[s,:] = np.array([samples[s].SlopyDurationDistn.a, samples[s].SlopyDurationDistn.b])

    one_trace_vals["times_samples"] = times_samples
    one_trace_vals["vals_samples"] = vals_samples
    one_trace_vals["u_samples"] = u_samples
    one_trace_vals["ch2_samples"] = ch2_samples
    one_trace_vals["sigma_samples"] = sigma_samples
    one_trace_vals["ab_samples"] = ab_samples

    trace_vals.append(one_trace_vals)

  all_results = dict(t: v for t, v in zip(trace_names, trace_vals)
  	  # trace_vals is a list/iterable of dicts. 
  	  # Each trace_vals dict has (key, value pairs):
  	  #    'params': analysis parameters, probably a dict?
  	  #    'data': numframesx2 array or equivalent
  	  #    'times_samples': num_samples+1 by num_slopey+1 array
  	  #    'vals_samples': num_samples+1 by num_slopey+1 array
  	  #    'u_samples': num_samples by 1 array
  	  #    'ch2_samples': num_samples by 2
  	  #    'sigma_samples': num_samples by 1
  	  #.   'ab_samples': global slopey distribution parameters, same values for all dicts in trace_vals
  
  
  
  scipy.io.savemat(outfile, all_results, long_field_names=True, oned_as='column')

