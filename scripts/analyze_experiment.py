#!/usr/bin/env python

from collections import namedtuple
import glob
import os.path
import sys

import numpy as np
import scipy.io
import yaml

from slopey2 import analyze
from slopey2.analyze import TraceData


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
  
  # TODO save samples

  # Samples is a list of SamplerStates, one for each iteration of the MCMC,
  # with the first element the initialization.
  # SamplerState is a named tuple composed of a named tuple, SlopeyDurationDistn,
  # which contains the a and b parameters that parameterize the gamma on slopeys, and
  # a dict of trace_name : TraceLatentVars named tuple key-value pairs.
  # TraceLatentVars contains a list ['times', 'red_vals', 'time_offset', 
  # 'green_channel_transform_slope', 'green_channel_transform_intercept', 'camera_noise_sigma']
  
  # My old matlab plotting code expects the results of saving a dict via savemat from
  # scipy.io. The dict was key-value pairs of tracenames and a dict
  #           {'params': params, 'data': data,
  #           'times_samples': times_samples, 'vals_samples': vals_samples,
  #           'u_samples': u_samples, 'ch2_samples': ch2_samples,
  #           'sigma_samples': sigma_samples} 
  # When loaded into matlab, the shapes of these dict values were:
  # times_samples: num_samples+1 by num_slopey+1 array
  # vals_samples: num_samples+1 by num_slopey array
  # ch2_samples: num_samples by 2 array
  # u_samples: num_samples by 1 vector of offsets
  # And data was num_frames by 2
  
  # Plan the first: re-create something similar so I can use my old matlab plotting code
  # without too much rewriting. The main difference will be each trace file will now 
  # also contain the a, b parameters for the slopey duration distribution (which will be the
  # same for all trace files in a directory, because they were analyzed together).
  
  # Replace with list comprehensions or equivalent after I get the form down:
  # I think I also need to import our named tuples or define them at the top?
  
  # make empty arrays for times_samples etc same shapes as in my original matlab code
  # including a list of trace_names
  
  for s in samples:
    # Iterate through samples and pull out the values for this iteration, dump
    # them in arrays.
    
    # do I also want to save the data like I originally did?
  
  all_results = dict(t: v for t, v in zip(trace_names, trace_vals)
  
  
  
  scipy.io.savemat(outfile, all_results, long_field_names=True, oned_as='column')

