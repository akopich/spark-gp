package org.apache.spark.ml.regression.benchmark

import breeze.linalg.{sum, DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.sin
import org.apache.spark.ml.commons.kernel.RBFKernel
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GaussianProcessRegression
import org.apache.spark.sql.SparkSession

import scala.util.Random

object PerformanceBenchmark extends App {
  val spark = SparkSession.builder()
    .appName("bench")
    .master(s"local[${args(0)}]").getOrCreate()
  import spark.sqlContext.implicits._

  val sampleSize = args(2).toInt
  val nFeatures = 3
  val parallelism = args(0).toInt * 4
  val expertSampleSize = args(1).toInt

  val instancesRDD = spark.sparkContext.parallelize(0 until parallelism).flatMap(index => {
    val random = new Random(13 * index)
    val X = BDM.create(sampleSize/parallelism,
      nFeatures,
      Array.fill(sampleSize * nFeatures/parallelism)(random.nextDouble()))
    val Y = sin(sum(X(*, ::)) / 1000d).toArray

     (0 until X.rows).map{ i=>
      val x = X(i, ::)
      val y = Y(i)
      LabeledPoint(y, Vectors.dense(x.t.toArray))
    }
  })

  val instances = instancesRDD.toDF.cache()
  instances.count()

  val gp = new GaussianProcessRegression()
    .setKernel(() => new RBFKernel(0.1))
    .setDatasetSizeForExpert(expertSampleSize)
    .setActiveSetSize(expertSampleSize)
    .setSeed(13)
    .setSigma2(1e-3)

  time(gp.fit(instances))


  def time[T](f: => T): T = {
    val start = System.currentTimeMillis()
    val result = f
    println("TIME: " + (System.currentTimeMillis() - start))
    result
  }
}
