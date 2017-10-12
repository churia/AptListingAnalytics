import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
//import org.apache.spark.SparkContext
//import org.apache.spark.SparkContext._
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel

//object classifier {
//	def main(args:Array[String]){
	// Load training data in LIBSVM format.
//		val sc = new SparkContext()

		val data = MLUtils.loadLibSVMFile(sc, "rent/train_data/part-*")
		//val newdata = MLUtils.loadLibSVMFile(sc, "rent/newdata/part-*")
	
		val high = data.filter({ case LabeledPoint(label, features) =>label==2})
		val med = data.filter({ case LabeledPoint(label, features) =>label==1})
		val low = data.filter({ case LabeledPoint(label, features) =>label==0})

		val low_sample = low.sample(false,0.55,123456)
		//val med_sample = med.sample(false,0.35,111111)
		//val high_new = med.union(high.map({case LabeledPoint(label, features)=>LabeledPoint(1.0, features)}))
		//val newdata = high.union(med_sample).union(low_sample)
		val newdata = high.union(high).union(high).union(high).union(high).union(high).union(med).union(med)union(low_sample)
		
		//MLUtils.saveAsLibSVMFile(newdata,"rent/new_data")
		
		val splits = newdata.randomSplit(Array(0.7, 0.3), seed = 11L)
		val trainingData = splits(0).cache()
		val testData = splits(1)
		

		val numClasses = 3
		val catMap = scala.collection.mutable.Map[Int, Int]((0,21),(2,12),(3,32),(4,24),(5,15),(6,9))
		val a = 0
		for (a<- 7 to 25){
			catMap.put(a,2)
		}
		val categoricalFeaturesInfo = catMap.toMap
		val numTrees = 500 // Use more in practice.
		val featureSubsetStrategy = "auto" // Let the algorithm choose.
		val impurity = "gini" //"variance"
		val maxDepth = 10
		val maxBins = 32

		// Run training algorithm to build the model
		val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
		  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

//		sc.stop()
		// Compute raw scores on the test set.
		val predictionAndLabels = testData.map { case LabeledPoint(label, features) =>
		  val prediction = model.predict(features)
		  (prediction, label)
		}

		// Get evaluation metrics.
		val metrics = new MulticlassMetrics(predictionAndLabels)
		metrics.precision
		metrics.confusionMatrix
		
		model.save(sc, "rent/RFmodel")
//	}
//}
