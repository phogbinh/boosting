#include <bits/stdc++.h>
#include "XLessClassifier.h"
#include "XMoreEqlClassifier.h"
#include "YLessClassifier.h"
#include "YMoreEqlClassifier.h"
#include "BoostingClassifier.h"

using namespace std;

#define DATA_FILENAME "data/basic.txt"
#define MAX_DOUBLE_DIGITS_N 20
#define MIN_X -5
#define MIN_Y -5
#define MAX_ITERATIONS_N 10000
#define OUTPUT_ROWS_N 1000
#define OUTPUT_COLS_N 1000

struct Sample {
  double x = 0.0;
  double y = 0.0;
  int r = 0;
};

vector<Sample> samples;
vector<double> weights;
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

double getError(Classifier* classifierPtr) {
  double error = 0.0;
  for (int i = 0; i < samples.size(); ++i)
    if (classifierPtr->classify(samples[i].x, samples[i].y) != samples[i].r) error += weights[i];
  return error;
}

void getMinErrorClassifierPtr(Classifier*& minErrorClassifierPtr, double& minError) {
  minErrorClassifierPtr = nullptr;
  minError = INT_MAX;
  for (Classifier* classifierPtr : classifierPtrs) {
    double error = getError(classifierPtr);
    if (error < minError) {
      minErrorClassifierPtr = classifierPtr;
      minError = error;
    }
  }
}

void updateWeights(Classifier* classifierPtr, double error) {
  for (int i = 0; i < samples.size(); ++i) {
    if (classifierPtr->classify(samples[i].x, samples[i].y) == samples[i].r) weights[i] /= 2.0 * (1.0 - error);
    else weights[i] /= 2.0 * error;
  }
}

int getErrorCount(const BoostingClassifier& boostingClassifier) {
  int errorCount = 0;
  for (int i = 0; i < samples.size(); ++i)
    if (boostingClassifier.classify(samples[i].x, samples[i].y) != samples[i].r) ++errorCount;
  return errorCount;
}

void output(const BoostingClassifier& boostingClassifier) {
  double xMin = INT_MAX;
  double xMax = INT_MIN;
  double yMin = INT_MAX;
  double yMax = INT_MIN;
  for (int i = 0; i < samples.size(); ++i) {
    if (samples[i].x < xMin) xMin = samples[i].x;
    if (samples[i].x > xMax) xMax = samples[i].x;
    if (samples[i].y < yMin) yMin = samples[i].y;
    if (samples[i].y > yMax) yMax = samples[i].y;
  }
  xMin -= 1;
  xMax += 1;
  yMin -= 1;
  yMax += 1;
  vector<double> ys(OUTPUT_ROWS_N);
  for (int i = 0; i < OUTPUT_ROWS_N; ++i) ys[i] = yMin + (yMax - yMin) * i / (OUTPUT_ROWS_N - 1.0);
  vector<double> xs(OUTPUT_COLS_N);
  for (int i = 0; i < OUTPUT_COLS_N; ++i) xs[i] = xMin + (xMax - xMin) * i / (OUTPUT_COLS_N - 1.0);
  freopen("f.txt", "w", stdout);
  for (int i = 0; i < OUTPUT_ROWS_N; ++i) {
    for (int j = 0; j < OUTPUT_COLS_N; ++j) printf("%f ", boostingClassifier.value(xs[j], ys[i]));
    printf("\n");
  }
}

void deleteClassifiers() {
  for (int i = 0; i < classifierPtrs.size(); ++i) delete classifierPtrs[i];
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
  weights.resize(samples.size(), 1.0 / (double)samples.size());
  BoostingClassifier boostingClassifier;
  int m = 0;
  while (true) {
    Classifier* classifierPtr;
    double error;
    getMinErrorClassifierPtr(classifierPtr, error);
    if (error == 0.0) boostingClassifier.add(classifierPtr, 1.0);
    else boostingClassifier.add(classifierPtr, log(1.0 / error - 1.0) / 2.0);
    if (getErrorCount(boostingClassifier) == 0) break;
    if (m == MAX_ITERATIONS_N - 1) break;
    updateWeights(classifierPtr, error);
    ++m;
  }
  printf("step indexed #%d: error rate = %d/%d, classifiers count = %d\n", m, getErrorCount(boostingClassifier), samples.size(), boostingClassifier.getClassifiersCount());
  output(boostingClassifier);
  deleteClassifiers();
  return 0;
}
