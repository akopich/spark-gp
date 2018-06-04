package org.apache.spark.ml.regression

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.optimize.LBFGSB
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.commons._
import org.apache.spark.ml.commons.kernel.Kernel
import org.apache.spark.ml.commons.util._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{Identifiable, Instrumentation}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

/**
  * Gaussian Process Regression.
  *
  * Fitting of hyperparameters and prediction for GPR is infeasible for large datasets due to
  * high computational complexity O(N^3^).
  *
  * This implementation relies on the Bayesian Committee Machine proposed in [2] for fitting and on
  * Projected Process Approximation for prediction Chapter 8.3.4 [1].
  *
  * This way the linear complexity in sample size is achieved for fitting,
  * while prediction complexity doesn't depend on it.
  *
  * [1] Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian Processes for Machine Learning
  * (Adaptive Computation and Machine Learning). The MIT Press.
  *
  * [2] Marc Peter Deisenroth and Jun Wei Ng. 2015. Distributed Gaussian processes.
  * In Proceedings of the 32nd International Conference on International Conference on Machine Learning
  * Volume 37 (ICML'15), Francis Bach and David Blei (Eds.), Vol. 37. JMLR.org 1481-1490.
  *
  */
class GaussianProcessRegression(override val uid: String)
  extends Regressor[Vector, GaussianProcessRegression, GaussianProcessRegressionModel]
    with GaussianProcessParams with ProjectedGaussianProcessHelper with GaussianProcessCommons with Logging {

  def this() = this(Identifiable.randomUID("gaussProcessReg"))

  override protected def train(dataset: Dataset[_]): GaussianProcessRegressionModel = {
    val instr = Instrumentation.create(this, dataset)

    val points: RDD[LabeledPoint] = getPoints(dataset).cache()

    val expertLabelsAndKernels: RDD[(BDV[Double], Kernel)] = getExpertLabelsAndKernels(points).cache()

    instr.log("Optimising the kernel hyperparameters")
    val optimalHyperparameters = optimizeHyperparameters(expertLabelsAndKernels)
    val optimalKernel = getKernel().setHyperparameters(optimalHyperparameters)
    instr.log("Optimal kernel: " + optimalKernel)

    expertLabelsAndKernels.foreach(_._2.setHyperparameters(optimalHyperparameters))

    val rawPredictor = projectedProcess(expertLabelsAndKernels, points, optimalHyperparameters, optimalKernel)
    val model = new GaussianProcessRegressionModel(uid, rawPredictor)
    instr.logSuccess(model)
    model
  }

  private def optimizeHyperparameters(expertLabelsAndKernels: RDD[(BDV[Double], Kernel)]) = {
    val f = new DiffFunctionMemoized[BDV[Double]] with Serializable {
      override protected def calculateNoMemory(x: BDV[Double]): (Double, BDV[Double]) = {
        expertLabelsAndKernels.treeAggregate((0d, BDV.zeros[Double](x.length)))({ case (u, (y, k)) =>
          k.setHyperparameters(x)
          val (likelihood, gradient) = likelihoodAndGradient(y, k)
          (u._1 + likelihood, u._2 += gradient)
        }, { case (u, v) =>
          (u._1 + v._1, u._2 += v._2)
        })
      }
    }

    val x0 = getKernel().getHyperparameters
    val (lower, upper) = getKernel().hyperparameterBoundaries
    val solver = new LBFGSB(lower, upper, maxIter = $(maxIter), tolerance = $(tol))

    solver.minimize(f, x0)
  }

  private def likelihoodAndGradient(y: BDV[Double], kernel : Kernel) = {
    val (k, derivative) = kernel.trainingKernelAndDerivative()
    val (_, logdet, kinv) = logDetAndInv(k)
    val alpha = kinv * y
    val likelihood = 0.5 * (y.t * alpha) + 0.5 * logdet

    val alphaAlphaTMinusKinv = alpha * alpha.t
    alphaAlphaTMinusKinv -= kinv

    val gradient = derivative.map(derivative => -0.5 * sum(derivative *= alphaAlphaTMinusKinv))
    (likelihood, BDV(gradient:_*))
  }

  override def copy(extra: ParamMap): GaussianProcessRegression = defaultCopy(extra)
}

class GaussianProcessRegressionModel private[regression](override val uid: String,
          private val gaussianProjectedProcessRawPredictor: GaussianProjectedProcessRawPredictor)
  extends RegressionModel[Vector, GaussianProcessRegressionModel] {

  override protected def predict(features: Vector): Double = {
    gaussianProjectedProcessRawPredictor.predict(features)
  }

  override def copy(extra: ParamMap): GaussianProcessRegressionModel = {
    val newModel = copyValues(new GaussianProcessRegressionModel(uid, gaussianProjectedProcessRawPredictor), extra)
    newModel.setParent(parent)
  }
}




