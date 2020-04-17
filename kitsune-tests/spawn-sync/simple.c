#include <stdio.h>
#include <kitsune.h>

void f(int n) {
  for(int i = 0; i < n; i++) spawn x {
      printf("%d\n", i);
    }
  sync x;
}

int main(int argc, char *argv[]) {
  f(10);
  return 0;
}

