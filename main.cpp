#include <bits/stdc++.h>

using namespace std;

#define DATA_FILENAME "data.txt"
#define MAX_DOUBLE_DIGITS_N 20

struct Sample {
  double x = 0.0;
  double y = 0.0;
  int r = 0;
};

vector<Sample> samples;

int main() {
  freopen(DATA_FILENAME, "r", stdin);
  char strX[MAX_DOUBLE_DIGITS_N];
  char strY[MAX_DOUBLE_DIGITS_N];
  int r;
  while (scanf("%[^,],%[^,],%d", strX, strY, &r) != EOF) {
    Sample sample;
    sample.x = stod(strX);
    sample.y = stod(strY);
    sample.r = r;
    samples.push_back(sample);
  }
  return 0;
}
