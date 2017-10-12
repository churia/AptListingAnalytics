import scala.util.parsing.json.JSON
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

//---------------apt info----------------
//get bathroom feature
val bathrooms = sc.wholeTextFiles("rent/test/test_bathrooms").
				map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
				flatMap(map=>map.toList)

//get bedroom feature
val bedrooms = sc.wholeTextFiles("rent/test/test_bedrooms").
				map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
				flatMap(map=>map.toList)

//get facilities features
val facilities = sc.wholeTextFiles("rent/test/test_features").
				map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,List[String]]]).
				flatMap(map=>map.toList)

val num_facility = facilities.mapValues(_.size)

val facility_list = sc.textFile("rent/facility_list").collect.toList

def hasFacility(fac:List[String], feats: List[String]):List[Double]={
	var fea = List[Double]()
	for (fc <- fac){
		if (feats.map(_.toLowerCase.replace("-","")).contains(fc)){
			fea = fea:+1.0
		}
		else{
			fea = fea:+0.0
		}
	}
	return fea
}

val facility_features = facilities.mapValues(line=>hasFacility(facility_list,line))

//get price
val price= sc.wholeTextFiles("rent/test/test_price").
				map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
				flatMap(map=>map.toList)

//get price/area = price/(1+N_bed + 0.5 * N_bath)
def Nbed(v:Double):Double ={
	if (v>4) return 4
	if (v<1) return 1
	return v
} 

def Nbath(v:Double):Double={
	if (v>2) return 2
	return v
}
val price_unit = bedrooms.join(bathrooms).mapValues(tp=>List(tp._1,tp._2)).join(price).
				mapValues(tp=>tp._2/(1+Nbed(tp._1(0))+0.5*Nbath(tp._1(1))))


//get description
val description = sc.wholeTextFiles("rent/test/test_description").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)
val description_length = description.mapValues(_.split("[\\p{Punct}\\s]+").size)

//get #photo
val num_photos = sc.wholeTextFiles("rent/test/test_photos").
				map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,List[String]]]).
				flatMap(map=>map.toList).map(tp=>(tp._1,tp._2.size))

//------------- geo info------------------
//get display_address
val display_address = sc.wholeTextFiles("rent/test/test_display_address").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)

//get street_address
val street_address = sc.wholeTextFiles("rent/test/test_street_address").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)

//get latitude & longitude
val latitude = sc.wholeTextFiles("rent/test/test_latitude").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
					flatMap(map=>map.toList)

val longitude = sc.wholeTextFiles("rent/test/test_longitude").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
					flatMap(map=>map.toList)

val Point = longitude.join(latitude)

val clusteringModel = KMeansModel.load(sc, "rent/KMeansModel")
val K = 20
def get_neighborhood(p:(Double,Double), cluster:KMeansModel): Int={
	val x = p._1
	val y = p._2
	if (x > -74.05 && x < -73.75 && y > 40.55 && y < 40.95){
		cluster.predict(Vectors.dense(x,y))
	}
	else return K
}
val neighborhood = Point.mapValues(p=>get_neighborhood(p,clusteringModel))

//get building_id
val building_id = sc.wholeTextFiles("rent/test/test_building_id").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)


val bscore = sc.textFile("rent/building_score").map(_.replace("(","").replace(")","").split(",")).map(l=>(l(0),l(1).toDouble))
bscore.cache()
val bsum = bscore.map(tp=>tp._2).reduce(_+_)
val bmeanscore = bsum/bscore.count()
val bscoreMap = bscore.collect.toMap
val building_quality = building_id.mapValues(x=>bscoreMap.getOrElse(x,bmeanscore))

//--------------listing info--------------
//get listing_id					
val listing_id = sc.wholeTextFiles("rent/test/test_listing_id").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
					flatMap(map=>map.toList)

//get train_manager_id
val manager_id = sc.wholeTextFiles("rent/test/test_manager_id").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)


val maga_code = sc.textFile("rent/manager_code").map(_.replace("(","").replace(")","").split(",")).map(l=>(l(0),l(1).toInt)).collect.toMap

def getMagaCode(id: String, encode:Map[String,Int]): Int = {
	encode.getOrElse(id, -1)
}
val manager_feature = manager_id.mapValues(x=>getMagaCode(x,maga_code).toDouble)

val maga_score = sc.textFile("rent/manager_score").map(_.replace("(","").replace(")","").split(",")).map(l=>(l(0),l(1).toDouble))
maga_score.cache()
val scoresum = maga_score.map(tp=>tp._2).reduce(_+_)
val meanscore = scoresum/maga_score.count()
val scoreMap = maga_score.collect.toMap

val manager_skill = manager_id.mapValues(x=>scoreMap.getOrElse(x,meanscore))

//get created time
val created = sc.wholeTextFiles("rent/test/test_created").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)

val created_hour = created.mapValues(_.split(" ")(1).split(":")(0).toDouble)
val created_day = created.mapValues(_.split(" ")(0).split("-")(2).toDouble)
val created_month = created.mapValues(_.split(" ")(0).split("-")(1).toDouble)

//---------------------------------------------------

//join feature
val features = neighborhood.join(manager_feature).mapValues(tp=>List(tp._1.toDouble,tp._2)).//21
					join(created_month).mapValues(tp=>tp._1:+tp._2).//12
					join(created_day).mapValues(tp=>tp._1:+tp._2).//31
					join(created_hour).mapValues(tp=>tp._1:+tp._2).//24
					join(bathrooms).mapValues(tp=>tp._1:+tp._2).//15
					join(bedrooms).mapValues(tp=>tp._1:+tp._2). //9
					join(facility_features).mapValues(tp=>tp._1++tp._2).
					join(num_facility).mapValues(tp=>tp._1:+tp._2.toDouble).
					join(price).mapValues(tp=>tp._1:+tp._2).
					join(price_unit).mapValues(tp=>tp._1:+tp._2).
					join(description_length).mapValues(tp=>tp._1:+tp._2.toDouble).
					join(num_photos).mapValues(tp=>tp._1:+tp._2.toDouble).
					join(listing_id).mapValues(tp=>tp._1:+tp._2).
					join(manager_skill).mapValues(tp=>tp._1:+tp._2).
					join(building_quality).mapValues(tp=>tp._1:+tp._2)

val test_data = features.map(X=>(0,X._2)).map(tp=>LabeledPoint(tp._1.toDouble, Vectors.dense(tp._2.toArray).toSparse))

MLUtils.saveAsLibSVMFile(test_data,"rent/test_data")
