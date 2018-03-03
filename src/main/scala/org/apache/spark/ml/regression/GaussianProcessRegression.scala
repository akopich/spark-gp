package org.apache.spark.ml.regression

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.regression.kernel.{Kernel, RBFKernel}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.col
import breeze.linalg.{inv, logdet, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.optimize.{DiffFunction, LBFGS, LBFGSB}
import breeze.linalg._
import breeze.numerics._
import breeze.math._

private[regression] trait GaussianProcessRegressionParams extends PredictorParams
  with HasMaxIter with HasTol with HasStandardization with HasAggregationDepth {

  final val kernel = new Param[() => Kernel](this,
    "kernel", "function of no arguments which returns " +
      "the kernel of the prior Gaussian Process")

  final val datasetSizeForExpert = new IntParam(this,
    "datasetSizeForExpert", "the number of data points fed to each expert")

  def setDatasetSizeForExpert(value: Int): this.type = set(datasetSizeForExpert, value)
  setDefault(datasetSizeForExpert -> 100)

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setKernel(value: () => Kernel): this.type = set(kernel, value)
  setDefault(kernel -> (() => new RBFKernel()))

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)
}

class GaussianProcessRegression(override val uid: String)
  extends Regressor[Vector, GaussianProcessRegression, GaussianProcessRegressionModel]
    with GaussianProcessRegressionParams {

  def this() = this(Identifiable.randomUID("gaussProcessReg"))

  override protected def train(dataset: Dataset[_]): GaussianProcessRegressionModel = {
    val points: RDD[LabeledPoint] = dataset.select(
      col($(labelCol)), col($(featuresCol))).rdd.map {
      case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features)
    }

    val groupedInstances = groupForExperts(points).cache()

    val expertLabels = groupedInstances.map(points => BDV(points.map(_.label).toSeq :_*))
    val expertKernels = groupedInstances.map(chunk =>
      $(kernel)().setTrainingVectors(chunk.map(_.features).toArray)
    )

    val expertLabelsAndKernels: RDD[(BDV[Double], Kernel)] = (expertLabels zip expertKernels).cache()

    val f = new DiffFunction[BDV[Double]] with Serializable {
      override def calculate(x: BDV[Double]): (Double, BDV[Double]) = {
        expertLabelsAndKernels.map { case (y, kernel) =>
          kernel.hyperparameters = Vectors.dense(x.toArray)
          likelihoodAndGradient(y, kernel)
        }.reduce { case ((l1, r1), (l2,r2)) => (l1+l2, r1+r2)}
      }
    }

    val x0 = $(kernel)().hyperparameters
    val solver = new LBFGSB(BDV[Double](x0.toArray.map(_ => -inf)),
      BDV[Double](x0.toArray.map(_ => inf)), maxIter = $(maxIter), tolerance = $(tol))

    val optimalHyperparameters = solver.minimize(f, BDV[Double](x0.toArray:_*))

    expertLabelsAndKernels.foreach { case(_, k) =>
      k.hyperparameters = Vectors.dense(optimalHyperparameters.toArray) }

    new GaussianProcessRegressionModel(uid, expertLabelsAndKernels)
  }

  private def likelihoodAndGradient(y: BDV[Double], kernel : Kernel) = {
    val (k, derivative) = kernel.trainingKernelAndDerivative()
    val Kinv = inv(k)
    val alpha = Kinv * y
    val firstTerm :BDV[Double] = 0.5 * y.t * Kinv * y
    val likelihood = firstTerm(0) - 0.5 * logdet(Kinv)._2
    val gradient = derivative.map(derivative => sum(-0.5 * (alpha * alpha.t - Kinv) *:* derivative))
    (likelihood, BDV(gradient:_*))
  }

  private def groupForExperts(points: RDD[LabeledPoint]) = {
    val numberOfExperts = Math.round(points.count().toDouble / $(datasetSizeForExpert))
    points.zipWithIndex.map { case(instance, index) =>
      (index % numberOfExperts, instance)
    }.groupByKey().map(_._2)
  }

  override def copy(extra: ParamMap): GaussianProcessRegression = ???
}

class GaussianProcessRegressionModel private[regression](override val uid: String,
     private val  expertLabelsAndKernels: RDD[(DenseVector[Double], Kernel)])
  extends RegressionModel[Vector, GaussianProcessRegressionModel] {

  override protected def predict(features: Vector): Double = ???

  override def copy(extra: ParamMap): GaussianProcessRegressionModel = ???
}