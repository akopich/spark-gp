name := "spark-gp"

version := "0.1"

scalaVersion := "2.12.13"

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.1.1"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.1.1"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.1.1"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.3" % Test
