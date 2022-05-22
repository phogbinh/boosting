class BoostingClassifier {
public:
  void add(Classifier* classifierPtr, double alpha) {
    if (_map.find(classifierPtr) == _map.end()) _map[classifierPtr] = alpha;
    else _map[classifierPtr] += alpha;
  }
  double value(double x, double y) {
    double weightedSum = 0.0;
    for (auto& element : _map) weightedSum += element.second * element.first->classify(x, y);
    return weightedSum;
  }
  int classify(double x, double y) {
    return value(x, y) >= 0.0 ? 1 : -1;
  }
  int getClassifiersCount() {
    return _map.size();
  }

private:
  std::unordered_map<Classifier*, double> _map; // caller must delete
};
