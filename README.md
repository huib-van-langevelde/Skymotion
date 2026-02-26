
Skymotion
=========

Cleaning up to support Paul Boven's first GJ3789 paper

In the Examples directory there is a simplpe example for creating data and then fitting.
The example with prlxf is simple parallax only fit.
Invoke with:

   python3 ../fitskym.py -r prlxf -gfs

Here 'prlxf' is the root for finding and creating data. Every run will need a skym-par-[root].yaml file to control the run (selecting model, priors, etc). If generating fake data, it uses a 'tru' file to pass parameters.  Output will be tagged with date and time.

