# spark-gp
Gaussian Process Regression and Classification on Apache Spark.

The thing works in linear time. 

Setting up a learner looks like
```scala
val gp = new GaussianProcessRegression()
  .setDatasetSizeForExpert(100)
  .setActiveSetSize(1000)
  .setSigma2(1e-4)
  .setKernel(() => 1 * new ARDRBFKernel(5) + 1.const * new EyeKernel)
```
Check out the [Regression](src/main/scala/org/apache/spark/ml/regression/examples/) and [Classification](src/main/scala/org/apache/spark/ml/classification/examples/) examples.
