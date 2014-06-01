#!/usr/bin/python
import commands, os
import numpy

#Model
fa = -3.0
fb = 3.5
mua = 0
mub = 0
comp = 1

#Simulation
dev = 0
block = 256
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
    out = 'dich'
    output = open('%s.dat' % out, 'w')
    jmax = 10
    for i in range(5,18):
        paths = 2**i
        if i < 8:
            block = paths
        else:
            block = 256
        s = 0.0
        for j in range(jmax):
            _cmd = './dich --dev=%d --fa=%s --fb=%s --mua=%s --mub=%s --comp=%d --block=%d --paths=%d --periods=%s --spp=%d --trans=%s --mode=%s --points=%d --beginx=%s --endx=%s --domain=%s --domainx=%s --logx=%d --samples=%d' % (dev, fa, fb, mua, mub, comp, block, paths, periods, spp, trans, mode, points, beginx, endx, domain, domainx, logx, samples)
            #print _cmd
            cmd = commands.getoutput(_cmd)
            #print cmd
            s += float(cmd)
        s = s/jmax
        print paths, s, points*paths*periods*spp/s/10**9
        print >>output, '%s %s %s' % (paths, s, points*paths*periods*spp/s/10**9)
    output.close()
    #os.system('tac %s.dat | gnuplot -e "fa=%s; fb=%s" physletta.plt' % (out,fa,fb))
    #os.system('mv -v %s.dat %s.png %s' % (out, out, DIRNAME))
