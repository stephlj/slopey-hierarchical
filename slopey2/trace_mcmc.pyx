# distutils: extra_compile_args = -O3 -w -ffast-math -mtune=native -march=native
# distutils: extra_link_args = -lm
# distutils: libraries = gsl
# distutils: include_dirs = /opt/local/include
# distutils: library_dirs = /opt/local/lib
# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

from collections import namedtuple

import numpy as np
cimport numpy as np

from libc.math cimport log, exp, log1p

TraceLatentVars = namedtuple(
    'TraceLatentVars',
    ['times', 'red_vals', 'time_offset', 'green_channel_transform_slope',
     'green_channel_transform_intercept', 'camera_noise_sigma'])

cdef extern from "gsl/gsl_sf_gamma.h":
    double gsl_sf_lngamma(double x)
    double gsl_sf_lnbeta(double a, double b)

cdef extern from "gsl/gsl_sf_psi.h":
    cdef double gsl_sf_psi(double x);    # digamma
    cdef double gsl_sf_psi_1(double x);  # trigamma

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T) nogil
    void gsl_rng_set(gsl_rng *r, unsigned long int s)

cdef extern from "gsl/gsl_randist.h":
    double gamma "gsl_ran_gamma"(gsl_rng * r, double, double)
    double beta "gsl_ran_beta"(gsl_rng * r, double, double)

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
gsl_rng_set(r, 0)  # should be seeded by env var, but this is just to make sure

cdef inline double dmin(double a, double b): return a if a < b else b
cdef inline double dmax(double a, double b): return a if a > b else b

cdef double PI = 3.141592653589793
cdef double EPS = 1e-8

cdef int NUM_SLOPEY_MAX = 25

cdef double[::1] _times = np.zeros(NUM_SLOPEY_MAX)
cdef double[::1] _new_times = np.zeros(NUM_SLOPEY_MAX)
cdef double[::1] _new_vals = np.zeros(NUM_SLOPEY_MAX//2+1)

cdef int NUM_FRAMES_MAX = 50000
cdef double[::1] _y_red = np.zeros(NUM_FRAMES_MAX)


def propose_local(dict config, trace_latents):
  # unpack variables we need
  cdef double time_scale = config['inference']['proposal_params']['trace']['time']
  cdef double ch2_scale = config['inference']['proposal_params']['ch2']
  cdef double sigma_scale = config['inference']['proposal_params']['sigma']
  cdef double[::1] raw_times = trace_latents.times
  cdef int k, K = raw_times.shape[0]
  cdef double[::1] vals = trace_latents.red_vals
  cdef double val_scale = config['inference']['proposal_params']['trace']['val']
  cdef double T_cycle = config['camera']['T_cycle']
  cdef double u_scale = config['inference']['proposal_params']['u']

  cdef double[::1] times = _times[:K]
  diff(raw_times, times)

  # propose new times (slopey start and end times)
  cdef double[::1] new_times = _new_times[:K]
  for k in range(0,K,2):
    new_times[k]   = dmax(EPS, gamma(r, times[k]   * time_scale, 1./time_scale))
    new_times[k+1] = dmax(EPS, gamma(r, times[k+1] * time_scale, 1./time_scale))
  cumsum(new_times)

  # propose new vals (red vals at start and end of slopey)
  cdef double[::1] new_vals = _new_vals[:(K//2+1)]
  for k in range(K//2+1):
    new_vals[k] = dmax(EPS, gamma(r, vals[k] * val_scale, 1./val_scale))

  # propose new ch2_transform_params
  cdef double new_slope, new_intercept
  new_slope = dmax(EPS, gamma(r, trace_latents.green_channel_transform_slope * ch2_scale, 1./ch2_scale))
  new_intercept = dmax(EPS, gamma(r, trace_latents.green_channel_transform_intercept * ch2_scale, 1./ch2_scale))

  # propose new sigma
  cdef double new_sigma
  new_sigma = dmax(EPS, gamma(r, trace_latents.camera_noise_sigma * sigma_scale, 1./sigma_scale))

  # propose new frame offset
  cdef double frac
  frac = trace_latents.time_offset / T_cycle
  cdef double new_time_offset = T_cycle * clip(beta(r, frac * u_scale, (1.-frac) * u_scale), EPS, 1.-EPS)

  new_trace_latents = TraceLatentVars(
      times = np.array(new_times, copy=True),
      red_vals = np.array(new_vals, copy=True),
      time_offset = new_time_offset,
      green_channel_transform_slope = new_slope,
      green_channel_transform_intercept = new_intercept,
      camera_noise_sigma = new_sigma)

  return new_trace_latents


cdef inline void diff(double[::1] a, double[::1] out):
  # like np.diff(np.concatenate(((0,), a)))
  cdef int k, K = a.shape[0]
  out[0] = a[0]
  for k in range(1, K):
    out[k] = a[k] - a[k-1]

cdef inline void cumsum(double[::1] a):
  # destructive cumsum just like np.cumsum
  cdef int k, K = a.shape[0]
  for k in range(1,K):
    a[k] += a[k-1]

cdef inline double clip(double x, double low, double high):
  return dmin(dmax(x, low), high)
