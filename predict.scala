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

val data = MLUtils.loadLibSVMFile(sc, "rent/test_data/part-*")

val model = RandomForestModel.load(sc, "rent/RFmodel")

val prediction = data.map(p=>model.predict(p.features))

prediction.saveAsTextFile("rent/prediction")

		
		

