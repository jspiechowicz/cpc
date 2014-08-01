set terminal pngcairo enhanced
set output 'test.png'
set key outside center top horizontal samplen 2
set autoscale fix
set logscale x 2
set logscale y
set format x '2^{%L}'
set format y '10^{%L}'
set xlabel 'N'
set ylabel 'Simulation time [s]' rotate by 90
p './poisson.dat' u 1:2 ti 'gpu float' w l lw 2, './cpucpoisson_i7.dat' u 1:($2/1000) ti 'cpu float' w l lw 2, './dpoisson.dat' u 1:2 ti 'gpu double' w l lw 2, './cpudcpoisson_i7.dat' u 1:($2/1000) ti 'cpu double' w l lw 2
exit gnuplot
