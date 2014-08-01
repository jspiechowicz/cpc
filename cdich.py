#!/usr/bin/python
import commands

#Model
fa = -3.0
fb = 3.5
comp = 1

#Simulation
paths = 2**17 #1024
periods = 1000
spp = 200
samples = 2000
trans = 0.1

#output
out = 'cdich'
output = open('%s.dat' % out, 'w')

#sim
def lim(tau): return fa/((fa-fb)*tau)

N = 10
A = -2 #1.54
B = 0 #46.2
step = (B-A)/float(N-1)
for tau in [10**(A + i*step) for i in range(N)]:
  mua = lim(tau)
  mub = -fb*mua/fa
  _cmd = './cdich --fa=%s --fb=%s --mua=%s --mub=%s --comp=%d --paths=%s --periods=%s --spp=%d --trans=%s --samples=%d' % (fa, fb, mua, mub, comp, paths, periods, spp, trans, samples)
  cmd = commands.getoutput(_cmd)
  print >>output, '#%s' % _cmd
  lcmd = []
  for l in cmd.split('\n'):
    if l[0] == '#':
      lcmd.append(l+" [5]mua")
    else:
      lcmd.append(l+" "+str(mua))
  cmd = '\n'.join(lcmd)
  print >>output, cmd
output.close()
