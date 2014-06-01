#!/usr/bin/python
import commands, os
import numpy

#Model
Dg = 0
Dp = 0
lmd = 0
fa = -3.0
fb = 3.5
mua = 0
mub = 0
comp = 1
mean = 0

#Simulation
dev = 0
block = 64
paths = 1024
periods = 1000
spp = 200
samples = 2000
trans = 0.1

#Output
mode = 'moments'
points = 100
beginx = 1.54
endx = 46.2
domain = '1d'
domainx = 'm'
logx = 0
DIRNAME='./'
#os.system('mkdir -p %s' % DIRNAME)
#os.system('rm -v %s*.dat %s*.png' % (DIRNAME, DIRNAME))

def lim(tau):
    return fa/((fa - fb)*tau)

for fb in [1.6]: #[1.6, 2.5, 3.5, 10]:
    tau = 0.01
    beginx = lim(tau)
    tau = 0.3
    endx = lim(tau)
    out = 'physletta_fb%s' % fb
    _cmd = './prog --dev=%d --Dg=%s --Dp=%s --lambda=%s --fa=%s --fb=%s --mua=%s --mub=%s --comp=%d --mean=%s --block=%d --paths=%d --periods=%s --spp=%d --trans=%s --mode=%s --points=%d --beginx=%s --endx=%s --domain=%s --domainx=%s --logx=%d --samples=%d >> %s.dat' % (dev, Dg, Dp, lmd, fa, fb, mua, mub, comp, mean, block, paths, periods, spp, trans, mode, points, beginx, endx, domain, domainx, logx, samples, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)
    os.system('tac %s.dat | gnuplot -e "fa=%s; fb=%s" physletta.plt' % (out,fa,fb))
    os.system('mv -v %s.dat %s.png %s' % (out, out, DIRNAME))
