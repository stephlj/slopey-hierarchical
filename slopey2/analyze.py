from collections import namedtuple

import numpy as np
import scipy.optimize


TraceData = namedtuple('TraceData', ['frames', 'guesses'])
SamplerState = namedtuple('SamplerState',
                          ['slopey_duration_distn', 'trace_latents'])
SlopeyDurationDistn = namedtuple('SlopeyDurationDistn', ['a', 'b'])
TraceLatentVars = namedtuple(
    'TraceLatentVars',
    ['times', 'red_vals', 'time_offset', 'green_channel_transform_slope',
     'green_channel_transform_intercept', 'camera_noise_sigma'])

def fit_green_transform(red_vals, green_vals, fudge=1e-2):
    flipped_red = np.max(red_vals) - red_vals
    data = np.vstack([flipped_red, np.ones_like(red_vals)]).T
    slope, intercept = scipy.optimize.nnls(data, green_vals)[0]
    return max(slope, fudge), max(intercept, fudge)

def initialize(traces, T_cycle):
  slopey_duration_distn = SlopeyDurationDistn(9./4, 3./4)  # TODO move elsewhere
  trace_latents = [initialize_trace_latents(trace_data, T_cycle)
                   for trace_name, trace_data in traces.items()]
  return SamplerState(slopey_duration_distn, trace_latents)

def initialize_trace_latents(trace_data, T_cycle):
  frames, frame_guesses = trace_data.frames, np.atleast_1d(trace_data.guesses)

  # initialize translocation times from guesses
  time_guesses = frame_guesses * T_cycle
  times = np.array([[t - 0.2, t + 0.2] for t in time_guesses]).ravel()
  if not np.all(times > 0) or not np.all(np.diff(times) > 0):
    raise ValueError(times)

  # initialize red vals by data averages within blocks
  idxs = [0] + list(frame_guesses) + [None]
  averages = [np.mean(frames[block_start:block_end], axis=0)
              for block_start, block_end in zip(idxs[:-1], idxs[1:])]
  averages = np.array(averages)
  red_vals = np.maximum(1e-3, averages[:, 0])

  # initialize transform from red ch to green ch
  green_vals = np.maximum(1e-3, averages[:, 1])
  slope, intercept = fit_green_transform(red_vals, green_vals)

  # set time offset to be 0.
  time_offset = 0.

  # set initial camera noise sigma to be a constant
  camera_noise_sigma = np.sqrt(2)

  return TraceLatentVars(times, red_vals, time_offset, slope, intercept,
                         camera_noise_sigma)


def run_mcmc(config, traces):
  z = initialize(traces, config['camera']['T_cycle'])
  # TODO
  import ipdb; ipdb.set_trace()
