#include "Classifier.h"

class YMoreEqlClassifier : public Classifier {
public:
  YMoreEqlClassifier(double threshold) : _threshold(threshold) {
    // body intentionally empty
  }
  virtual int classify(double x, double y) override {
    return y >= _threshold ? 1 : -1;
  }

private:
  double _threshold = 0;
};
