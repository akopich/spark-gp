# spark-gp
Gaussian Process Regression and Classification on Apache Spark
WIP

The thing works in linear time. 

Docs are yet to come, so check out examples `src/main/scala/org/apache/spark/ml/regression/examples/` and `src/main/scala/org/apache/spark/ml/classification/examples/`.


##  Writing the model: 

     val gp = new GaussianProcessRegression()
    .setActiveSetSize(1000)
    .setSigma2(1e-4)
    .fit(trainingSet)    
    gp.save(path)


##  Reading the model: 

    val model = new GaussianProcessRegressionModelReader().load(path)
    model.predict(vect)
