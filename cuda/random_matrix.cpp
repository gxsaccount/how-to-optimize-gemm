#include <stdlib.h>

#define A(i, j) a[(j)*lda + (i)]
extern double drand48();

void random_matrix(int m, int n, float *a, int lda) {
  double drand48();
  int i, j;
  int count=0;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
#if 1
      A(i, j) = 2.0 * (float)drand48() - 1.0;
#else
      // A(i, j) = (j - i) % 3;
      // A(i, j) = (count++);
      // A(i, j) = 1;
#endif
}
