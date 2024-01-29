<!--  0,  1,  2,  3,  4,  5,   6,  7 -->
<!-- x1, y1, z1, x2, y2, z2, 0.1,  1 -->

# 10
out = x2

# 79 
out = sin(exp(z1) * y2)

# 204
<!-- TODO -->

# 206
out = sin(abs(z2 > x1) * y2)

# 318
out = abs(z2 > x1) * y2

# 352
out = sqrt(z2 > x1) * y2

# 367 (best)
out = sqrt(x2 > z1) * y2

# 376
out = ((x2 > x1) / 1) * y2

# 573
out = (((1 * z2) < ((x2 - (1 * x1)) - cos(x2))) < (0.1 * ((1 < abs(z1)) < z2)))

# 626
out = sin(x2 < x1) * y2
