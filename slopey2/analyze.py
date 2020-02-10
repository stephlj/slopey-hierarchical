from collections import namedtuple

import numpy as np
import numpy.random as npr
import scipy.optimize

from slopey2.trace_mcmc import propose_local, TraceLatentVars

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
  trace_latents = {trace_name : initialize_trace_latents(trace_data, T_cycle)
                   for trace_name, trace_data in traces.items()}
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

def propose(config, state):
  new_slopey_duration_distn = propose_global(config, state.slopey_duration_distn)
  new_trace_latents = {trace_name : propose_local(config, local_vars)
                       for trace_name, local_vars in state.trace_latents.items()}
  return SamplerState(new_slopey_duration_distn, new_trace_latents)

def propose_global(config, slopey_duration_distn):
  scale = config['inference']['proposal_params']['global']
  new_a = np.maximum(1e-6, npr.gamma(slopey_duration_distn.a * scale, 1. / scale))
  new_b = np.maximum(1e-6, npr.gamma(slopey_duration_distn.b * scale, 1. / scale))
  new_slopey_duration_distn = SlopeyDurationDistn(a=new_a, b=new_b)
  return new_slopey_duration_distn

# TODO local_log_proposal_diff in cython (from fast.pyx)
# TODO local_log_prior_diff in cython (from fast.pyx)
# TODO global_log_prior_diff in Python (from priors.py)
# TODO global_log_proposal_diff in Python (from proposals.py)

def log_accept_prob(config, state, new_state, traces):
  global_diff = (global_log_prior_diff(config, state.globals, new_state.globals) +
                 global_log_proposal_diff(config, state.globals, new_state.globals))
  locals_diff = sum([
      local_log_proposal_diff(config,
                              state.trace_latents[trace_name],
                              new_state.trace_latents[trace_name])
      + local_log_prior_diff(config,
                             state.trace_latents[trace_name],
                             new_state.trace_latents[trace_name])
      + camera_loglike(config, state.trace_latents[trace_name],     traces[trace_name])
      - camera_loglike(config, new_state.trace_latents[trace_name], traces[trace_name])
      for trace_name in state.trace_latents.keys()])
  return global_diff + locals_diff

def flip_coin(p):
  return npr.uniform() < p

def mh_step(config, state, traces):
  new_state = propose(config, state)
  accept_prob = np.exp(log_accept_prob(config, state, new_state, traces))
  accept = flip_coin(accept_prob)
  return new_state if accept else state

def run_mcmc(config, traces):
  z = initialize(traces, config['camera']['T_cycle'])
  samples = [z]
  for itr in range(config['inference']['num_iterations']):
    z = mh_step(config, z, traces)
    samples.append(z)

  return samples
