#include "Classifier.h"

class XMoreEqlClassifier : public Classifier {
public:
  XMoreEqlClassifier(double threshold) : _threshold(threshold) {
    // body intentionally empty
  }
  virtual int classify(double x, double y) override {
    return x >= _threshold ? 1 : -1;
  }

private:
  double _threshold = 0;
};
