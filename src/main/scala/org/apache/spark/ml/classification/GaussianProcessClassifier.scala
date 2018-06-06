package org.apache.spark.ml.classification

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics
import breeze.numerics.{abs, exp, sigmoid, sqrt}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.commons._
import org.apache.spark.ml.commons.kernel.Kernel
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{Identifiable, Instrumentation}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

/**
  * Gaussian Process Classifier.
  *
  * Fitting of hyperparameters and prediction for GPR is infeasible for large datasets due to
  * high computational complexity O(N^3^).
  *
  * This implementation relies on the Bayesian Committee Machine proposed in [2] for fitting and on
  * Projected Process Approximation for prediction Chapter 8.3.4 [1].
  *
  * The classifier uses sigmoid link function which makes posterior intractable. The issue is overcome via
  * Laplace approximation in a manner similar to the one presented in Chapters 3 and 5 [1].
  *
  * This way the linear complexity in sample size is achieved for fitting,
  * while prediction complexity doesn't depend on it.
  *
  * This implementation supports binary classification only.
  *
  * [1] Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian Processes for Machine Learning
  * (Adaptive Computation and Machine Learning). The MIT Press.
  *
  * [2] Marc Peter Deisenroth and Jun Wei Ng. 2015. Distributed Gaussian processes.
  * In Proceedings of the 32nd International Conference on International Conference on Machine Learning
  * Volume 37 (ICML'15), Francis Bach and David Blei (Eds.), Vol. 37. JMLR.org 1481-1490.
  *
  */
class GaussianProcessClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, GaussianProcessClassifier, GaussianProcessClassificationModel]
    with GaussianProcessParams
    with GaussianProcessCommons[Vector, GaussianProcessClassifier, GaussianProcessClassificationModel]
    with Logging {
  def this() = this(Identifiable.randomUID("gaussProcessClass"))

  override protected def train(dataset: Dataset[_]): GaussianProcessClassificationModel = {
    val instr = Instrumentation.create(this, dataset)
    val points: RDD[LabeledPoint] = getPoints(dataset)

    // RDD of (y, f, kernel)
    val expertLabelsHiddensAndKernels: RDD[(BDV[Double], BDV[Double], Kernel)] = getExpertLabelsAndKernels(points)
      .map {case(y, kernel) => (y, BDV.zeros[Double](y.size), kernel)}
      .cache()

    val optimalHyperparameters = optimizeHypers(instr, expertLabelsHiddensAndKernels, likelihoodAndGradient)

    // Set the optimal hypers to all the kernels and estimate the corresponding f
    expertLabelsHiddensAndKernels.foreach(yfk =>  likelihoodAndGradient(yfk, optimalHyperparameters))

    produceModel(instr,
      points,
      expertLabelsHiddensAndKernels.map {case(_, f, kernel) => (f, kernel) },
      optimalHyperparameters)
  }

  private def likelihoodAndGradient(yFandK: (BDV[Double], BDV[Double], Kernel), x: BDV[Double]) = {
    val (y: BDV[Double], f: BDV[Double], kernel: Kernel) = yFandK
    kernel.setHyperparameters(x)
    val (kernelMatrix, derivatives) = kernel.trainingKernelAndDerivative()

    var oldObj = Double.NegativeInfinity
    var newObj = Double.MinValue

    val L = BDM.zeros[Double](y.length, y.length)
    val sqrtW = BDM.zeros[Double](y.length, y.length)
    val pi = BDV.zeros[Double](y.length)
    val a = BDV.zeros[Double](y.length)
    val gradLogP = BDV.zeros[Double](y.length)
    val I = BDM.eye[Double](y.length)
    var newtonStepSize = 1d

    //  The loop below (Algorithm 3.1 from [1]) is Newtonian optimization of q(f|X, y) w.r.t. f.
    while (abs(oldObj - newObj) > $(tol)) {
      pi := sigmoid(f)
      val W = diag(pi * (1d - pi)) // TODO optimize me
      sqrtW := sqrt(W)
      val B = diag(sqrtW) * diag(sqrtW).t
      B :*= kernelMatrix
      B += I // B = I + sqrtW * kernelMatrix * sqrtW
      L := cholesky(B)
      gradLogP := y - pi
      val b = W * f + gradLogP
      a := b - sqrtW * L.t \ (L \ (sqrtW * (kernelMatrix * b)))
      val newfCandidate = (1 - newtonStepSize) * f + newtonStepSize * kernelMatrix * a
      val newObjCandidate = -a.t * newfCandidate / 2d + sum(numerics.log(sigmoid((y * 2d - 1d) *:* newfCandidate)))
      if (newObjCandidate > oldObj) {
        f := newfCandidate
        oldObj = newObj
        newObj = newObjCandidate
      } else {
        newtonStepSize /= 2
      }
    }

    // below we calculate the approximating log-likelihood and its derivatives using Algorithm 5.1 from [1].
    val logZ = newObj - sum(numerics.log(diag(L)))

    val R = sqrtW * L.t \ (L \ sqrtW)
    val C = L \ (sqrtW * kernelMatrix)
    val d3logP = -(2d * pi - 1d) *:* pi *:* pi *:* exp(-f)
    val s2 = - 0.5 * (diag(kernelMatrix) - diag(C.t * C)) *:* d3logP

    val gradLogZ = BDV[Double](derivatives.map(C => {
      val s1 = 0.5 * (a.t * C * a) - 0.5 * sum(R *:* C)
      val b = C * gradLogP
      val s3 = b - kernelMatrix * R * b
      s1 + s2.t * s3
    }))

    (-logZ, -gradLogZ)
  }

  override def copy(extra: ParamMap): GaussianProcessClassifier = defaultCopy(extra)

  override protected def createModel(uid: String,
                                     rawPredictor: GaussianProjectedProcessRawPredictor): GaussianProcessClassificationModel = new GaussianProcessClassificationModel(uid, rawPredictor)
}

class GaussianProcessClassificationModel private[classification](override val uid: String,
              private val gaussianProjectedProcessRawPredictor: GaussianProjectedProcessRawPredictor)
  extends ProbabilisticClassificationModel[Vector, GaussianProcessClassificationModel] {

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = rawPrediction match {
    case dv : DenseVector =>
      dv.values(0) = sigmoid(-dv.values(0))
      dv.values(1) = 1 - dv.values(0)
      rawPrediction
    case _: SparseVector =>
      throw new RuntimeException("Unexpected error in GaussianProcessClassificationModel:" +
        " raw2probabilitiesInPlace encountered SparseVector")
  }

  override def numClasses: Int = 2

  override protected def predictRaw(features: Vector): Vector = {
    val (f, _) = gaussianProjectedProcessRawPredictor.predict(features)
    Vectors.dense(-f, f)
  }

  override def copy(extra: ParamMap): GaussianProcessClassificationModel = {
    val newModel = copyValues(new GaussianProcessClassificationModel(uid, gaussianProjectedProcessRawPredictor), extra)
    newModel.setParent(parent)
  }
}
