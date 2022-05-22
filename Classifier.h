#ifndef CLASSIFIER_H
#define CLASSIFIER_H
class Classifier {
public:
  virtual int classify(double x, double y) = 0;
};
#endif
