package org.apache.spark.ml.commons.util

import breeze.optimize.DiffFunction

import scala.collection.mutable


private[ml] trait DiffFunctionMemoized[T] extends DiffFunction[T] {
  override def calculate(x: T): (Double, T) = {
    cache.getOrElseUpdate(x, calculateNoMemory(x))
  }

  private val cache = mutable.HashMap[T, (Double, T)]()

  protected def calculateNoMemory(x: T): (Double, T)
}
