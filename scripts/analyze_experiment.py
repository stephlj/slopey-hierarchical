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

  # TODO save results

