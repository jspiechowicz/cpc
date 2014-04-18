set terminal pngcairo enhanced
set output sprintf('epl_mean%g.png',m)
set key outside center top horizontal samplen 2
set autoscale fix
set logscale x
set logscale y
set format x '10^{%L}'
set format y '10^{%L}'
set xlabel 'D_P'
set ylabel '{/Symbol \341}v{/Symbol \361}' rotate by 360

a(x) = sqrt(x*(m**2/x))
bp(x) = exp( 1.0/(x*(1.0 - 1.0/a(x))) ) 
bm(x) = exp( 1.0/(x*(1.0 + 1.0/a(x))) ) 
J(x) = (1.0/(4*x))*(bp(x) - bm(x))/((bp(x) - 1.0)*(bm(x) - 1.0))

p sprintf('epl_mean%g.dat',m) u 1:2 ti 'n' w l, 2*J(x) ti 'a' w l
exit gnuplot
