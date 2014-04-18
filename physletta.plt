set terminal pngcairo enhanced
set output sprintf('physletta_fb%g.png',fb)
set key outside center top horizontal samplen 2
set autoscale fix
set xlabel '{/Symbol t}'
set ylabel '{/Symbol \341}v{/Symbol \361}' rotate by 360

mub(x) = abs(fb*x/fa)
t(x) = 1.0/(x + mub(x))
bp(x) = exp(1.0/( (abs(fa) + 1)*(abs(fb) - 1)*t(x) ) )
bm(x) = exp(1.0/( (abs(fa) - 1)*(abs(fb) + 1)*t(x) ) )
J(x) = 1.0/( 4.0*abs(fa*fb)*t(x) )*( bm(x) - bp(x) )/( (bp(x) - 1.0)*(bm(x) - 1.0) )

p sprintf('physletta_fb%g.dat', fb) u (t($1)):2 ti 'n' w l, 2*J(t(x)) ti 'a' w l
exit gnuplot
