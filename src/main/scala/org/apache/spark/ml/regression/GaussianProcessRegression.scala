package org.apache.spark.ml.regression

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap}
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
  with HasMaxIter with HasTol with HasStandardization with HasAggregationDepth with HasSeed {

  final val kernel = new Param[() => Kernel](this,
    "kernel", "function of no arguments which returns " +
      "the kernel of the prior Gaussian Process")

  final val datasetSizeForExpert = new IntParam(this,
    "datasetSizeForExpert", "the number of data points fed to each expert")

  final val sigma2 = new DoubleParam(this,
    "sigma2", "The variance of noise in the inputs. The value is added to the diagonal of the " +
      "kernel Matrix. Also prevents numerical issues associated with inversion " +
      "of a computationally singular matrix ")

  final val activeSetSize = new IntParam(this,
    "activeSetSize", "number of latent functions to project the process onto")

  def setDatasetSizeForExpert(value: Int): this.type = set(datasetSizeForExpert, value)
  setDefault(datasetSizeForExpert -> 100)

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setSigma2(value: Double): this.type = set(sigma2, value)
  setDefault(sigma2 -> 1e-3)

  def setKernel(value: () => Kernel): this.type = set(kernel, value)
  setDefault(kernel -> (() => new RBFKernel()))

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  def setActiveSetSize(value: Int): this.type = set(activeSetSize, value)
  setDefault(activeSetSize -> 100)

  def setSeed(value: Long): this.type = set(seed, value)
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
    val sigma2 = $(this.sigma2)

    val f = new DiffFunction[BDV[Double]] with Serializable {
      override def calculate(x: BDV[Double]): (Double, BDV[Double]) = {
        expertLabelsAndKernels.map { case (y, kernel) =>
          kernel.hyperparameters = Vectors.dense(x.toArray)
          likelihoodAndGradient(y, kernel, sigma2)
        }.reduce { case ((l1, r1), (l2,r2)) => (l1+l2, r1+r2)}
      }
    }

    val x0 = $(kernel)().hyperparameters
    val solver = new LBFGSB(BDV[Double](x0.toArray.map(_ => -inf)),
      BDV[Double](x0.toArray.map(_ => inf)), maxIter = $(maxIter), tolerance = $(tol))

    val optimalHyperparameters = solver.minimize(f, BDV[Double](x0.toArray:_*))

    val activeSet = points.takeSample(withReplacement = false,
      $(activeSetSize), $(seed)).map(_.features)
    val activeSetBC = points.sparkContext.broadcast(activeSet)

    val KmnKnm2Kmny = expertLabelsAndKernels.map { case(y, k) =>
      k.hyperparameters = Vectors.dense(optimalHyperparameters.toArray)
      val kernelMatrix = k.crossKernel(activeSetBC.value)
      (kernelMatrix * kernelMatrix.t, kernelMatrix * y)
    }.reduce{ case ((l1, r1), (l2,r2)) => (l1+l2, r1+r2) }

    val magicKernel = $(kernel)().setTrainingVectors(activeSet)
    magicKernel.hyperparameters = Vectors.dense(optimalHyperparameters.toArray)

    val Kmm = regularizeMatrix(magicKernel.trainingKernel(), sigma2)

    val magicVector = inv(sigma2 * Kmm + KmnKnm2Kmny._1) * KmnKnm2Kmny._2

    new GaussianProcessRegressionModel(uid, magicVector, magicKernel)
  }

  private def likelihoodAndGradient(y: BDV[Double], kernel : Kernel, sigma2: Double) = {
    val (kNotRegularized, derivative) = kernel.trainingKernelAndDerivative()
    val k = regularizeMatrix(kNotRegularized, sigma2)
    val Kinv = inv(k)
    val alpha = Kinv * y
    val firstTerm :BDV[Double] = 0.5 * y.t * Kinv * y
    val likelihood = firstTerm(0) + 0.5 * logdet(k)._2
    val gradient = derivative.map(derivative => -0.5 * sum((alpha * alpha.t - Kinv) *:* derivative))
    (likelihood, BDV(gradient:_*))
  }

  private def groupForExperts(points: RDD[LabeledPoint]) = {
    val numberOfExperts = Math.round(points.count().toDouble / $(datasetSizeForExpert))
    points.zipWithIndex.map { case(instance, index) =>
      (index % numberOfExperts, instance)
    }.groupByKey().map(_._2)
  }

  private def regularizeMatrix(matrix : BDM[Double], regularization: Double) = {
    matrix + diag(BDV[Double]((0 until matrix.cols).map(_ => regularization).toArray))
  }

  override def copy(extra: ParamMap): GaussianProcessRegression = ???
}

class GaussianProcessRegressionModel private[regression](override val uid: String,
                                                         val magicVector: BDV[Double],
                                                         val kernel: Kernel)
  extends RegressionModel[Vector, GaussianProcessRegressionModel] {

  override def predict(features: Vector): Double = {
    val kstar = kernel.crossKernel(Array(features))
    val res = kstar * magicVector
    res(0)
  }

  override def copy(extra: ParamMap): GaussianProcessRegressionModel = ???
}