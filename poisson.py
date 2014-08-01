#!/usr/bin/python
import commands, os

#Model
Dp = 0
lmd = 0
mean = 2.5

#Simulation
dev = 0
block = 256
paths = 1024
periods = 1000
spp = 400
samples = 4000
trans = 0.1

#Output
mode = 'moments'
points = 100
beginx = -1
endx = 1
domain = '1d'
domainx = 'p'
logx = 1
DIRNAME='./'
#os.system('mkdir -p %s' % DIRNAME)
#os.system('rm -v %s*.dat %s*.png' % (DIRNAME, DIRNAME))

for mean in [2.5]: #[2.5, 5, 10]:
    out = 'poisson'
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
            _cmd = './poisson --dev=%d --Dp=%s --lambda=%s --mean=%s --block=%d --paths=%d --periods=%s --spp=%d --trans=%s --mode=%s --points=%d --beginx=%s --endx=%s --domain=%s --domainx=%s --logx=%d --samples=%d' % (dev, Dp, lmd, mean, block, paths, periods, spp, trans, mode, points, beginx, endx, domain, domainx, logx, samples)
            #print _cmd
            cmd = commands.getoutput(_cmd)
            #print cmd
	    s += float(cmd)
        s = s/jmax
        speed = points*paths*periods*spp/s/10**9
        print paths, s, speed
        print >>output, '%s %s %s' % (paths, s, speed)
    output.close()
    #os.system('gnuplot -e "m=%s" epl.plt' % mean)
    #os.system('mv -v %s.dat %s.png %s' % (out, out, DIRNAME))
