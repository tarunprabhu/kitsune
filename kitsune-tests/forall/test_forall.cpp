#include <cstdio>

int main() {
  forall(int i = 0; i < 5; ++i) {
    int j = i;
    printf("%d\n", i);
  }
  return 0;
}
