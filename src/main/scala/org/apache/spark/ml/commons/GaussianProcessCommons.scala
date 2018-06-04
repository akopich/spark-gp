package org.apache.spark.ml.commons

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.commons.kernel.{EyeKernel, Kernel, _}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Row}

trait GaussianProcessCommons extends ProjectedGaussianProcessHelper {
  this: Predictor[_, _, _] with GaussianProcessParams =>

  protected val getKernel : () => Kernel = () => $(kernel)() + $(sigma2).const * new EyeKernel

  protected def getPoints(dataset: Dataset[_]) = {
    dataset.select(col($(labelCol)), col($(featuresCol))).rdd.map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }
  }

  protected def groupForExperts(points: RDD[LabeledPoint]) = {
    val numberOfExperts = Math.round(points.count().toDouble / $(datasetSizeForExpert))
    points.zipWithIndex.map { case(instance, index) =>
      (index % numberOfExperts, instance)
    }.groupByKey().map(_._2)
  }

  protected def getExpertLabelsAndKernels(points: RDD[LabeledPoint]): RDD[(BDV[Double], Kernel)] = {
    groupForExperts(points).map { chunk =>
      val (labels, trainingVectors) = chunk.map(lp => (lp.label, lp.features)).toArray.unzip
      (BDV(labels: _*), getKernel().setTrainingVectors(trainingVectors))
    }
  }

  protected def projectedProcess(expertLabelsAndKernels: RDD[(BDV[Double], Kernel)],
                                 points: RDD[LabeledPoint],
                                 optimalHyperparameters: BDV[Double],
                                 optimalKernel: Kernel) = {
    val activeSet = $(activeSetProvider)($(activeSetSize), expertLabelsAndKernels, points,
      getKernel, optimalHyperparameters, $(seed))

    points.unpersist()

    val (matrixKmnKnm, vectorKmny) = getMatrixKmnKnmAndVectorKmny(expertLabelsAndKernels, activeSet)

    expertLabelsAndKernels.unpersist()

    optimalKernel.setTrainingVectors(activeSet)

    // inv(sigma^2 K_mm + K_mn * K_nm) * K_mn * y
    val magicVector = getMagicVector(optimalKernel, matrixKmnKnm, vectorKmny, activeSet, optimalHyperparameters)

    new GaussianProjectedProcessRawPredictor(magicVector, optimalKernel)
  }
}

class GaussianProjectedProcessRawPredictor private[commons] (val magicVector: BDV[Double],
                                                          val kernel: Kernel) extends Serializable {
  def predict(features: Vector): Double = {
    kernel.crossKernel(features) * magicVector
  }
}
