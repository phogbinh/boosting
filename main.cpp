#include <bits/stdc++.h>
#include "XLessClassifier.h"
#include "XMoreEqlClassifier.h"
#include "YLessClassifier.h"
#include "YMoreEqlClassifier.h"

using namespace std;

#define DATA_FILENAME "data.txt"
#define MAX_DOUBLE_DIGITS_N 20
#define MIN_X -5
#define MIN_Y -5

struct Sample {
  double x = 0.0;
  double y = 0.0;
  int r = 0;
};

vector<Sample> samples;
vector<Classifier*> classifierPtrs;

void constructDecisionTreeStumps() {
  vector<double> xs(samples.size());
  vector<double> ys(samples.size());
  for (int i = 0; i < samples.size(); ++i) {
    xs[i] = samples[i].x;
    ys[i] = samples[i].y;
  }
  sort(xs.begin(), xs.end());
  sort(ys.begin(), ys.end());
  {
    double prevX = MIN_X;
    for (auto x : xs) {
      double threshold = (prevX + x) / 2.0;
      classifierPtrs.push_back(new XLessClassifier(threshold));
      classifierPtrs.push_back(new XMoreEqlClassifier(threshold));
      prevX = x;
    }
  }
  {
    double prevY = MIN_Y;
    for (auto y : ys) {
      double threshold = (prevY + y) / 2.0;
      classifierPtrs.push_back(new YLessClassifier(threshold));
      classifierPtrs.push_back(new YMoreEqlClassifier(threshold));
      prevY = y;
    }
  }
}

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
  constructDecisionTreeStumps();
  return 0;
}
