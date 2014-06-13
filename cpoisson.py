#!/usr/bin/python
import commands, os
import numpy

#Model
mean = 2.5

#Simulation
paths = 2**17 #1024
periods = 1000
spp = 400
trans = 0.1
samples = 4000

N = 10
A = -1
B = 1
step = (B-A)/float(N-1)
for Dp in [10**(A+i*step) for i in range(N)]: #[0.1, 0.3, 0.5, 0.8, 1]:
  lmd = mean*mean/Dp
  out = 'cpoisson_i7'
  _cmd = './cpoisson --Dp=%s --lambda=%s --mean=%s --paths=%d --periods=%s --spp=%d --trans=%s --samples=%s >> %s.dat' % (Dp, lmd, mean, paths, periods, spp, trans, samples, out)
  output = open('%s.dat' % out, 'a')
  print >>output, '#%s' % _cmd
  output.close()
  print _cmd
  cmd = commands.getoutput(_cmd)
  #os.system('gnuplot -e "m=%s" epl.plt' % mean)
  #os.system('mv -v %s.dat %s.png %s' % (out, out, DIRNAME))
