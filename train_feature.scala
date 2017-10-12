import scala.util.parsing.json.JSON
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.{HashingTF, IDF}

def getLabelLevel(label:String):Double = {
	if (label == "low") return 0.0
	else if (label == "high") return 2.0
	else return 1.0
}

//get label
val train_label = sc.wholeTextFiles("rent/train/train_interest_level").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList).mapValues(x=>getLabelLevel(x))

//---------------apt info----------------
//get bathroom feature
val bathrooms = sc.wholeTextFiles("rent/train/train_bathrooms").
				map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
				flatMap(map=>map.toList)

//get bedroom feature
val bedrooms = sc.wholeTextFiles("rent/train/train_bedrooms").
				map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
				flatMap(map=>map.toList)

//get facilities features
val facilities = sc.wholeTextFiles("rent/train/train_features").
				map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,List[String]]]).
				flatMap(map=>map.toList)

facilities.cache()
val num_facility = facilities.mapValues(_.size)

val facility_type = facilities.flatMap(tp=>tp._2).
					map(x=>(x.toLowerCase.replace("-",""),1)).
					reduceByKey(_+_).
					map(tp=>(tp._2,tp._1)).
					sortByKey(false)

val top = 20
val facility_list = facility_type.take(top).map(tp=>tp._2).toList
sc.parallelize(facility_list).saveAsTextFile("rent/facility_list")

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
//for double check
//facility_type.saveAsTextFile("rent/facility.txt")

//get price
val price= sc.wholeTextFiles("rent/train/train_price").
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
val description = sc.wholeTextFiles("rent/train/train_description").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)
val description_length = description.mapValues(_.split("[\\p{Punct}\\s]+").size)

//val description_text = description.mapValues(_.split("[\\p{Punct}\\s]+".toSeq))

//val hashingTF = new HashingTF()
//val tf = hashingTF.transform(description_text)
//tf.cache()
//val idf = new IDF().fit(tf)
//val tfidf = idf.transform(tf)
//val description_TFIDF = description.map(tp=>tp._1).zipWithIndex(tfidf)

//get #photo
val num_photos = sc.wholeTextFiles("rent/train/train_photos").
				map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,List[String]]]).
				flatMap(map=>map.toList).map(tp=>(tp._1,tp._2.size))

//------------- geo info------------------
//get display_address
val display_address = sc.wholeTextFiles("rent/train/train_display_address").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)

//get street_address
val street_address = sc.wholeTextFiles("rent/train/train_street_address").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)

//get latitude & longitude
val latitude = sc.wholeTextFiles("rent/train/train_latitude").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
					flatMap(map=>map.toList)

val longitude = sc.wholeTextFiles("rent/train/train_longitude").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
					flatMap(map=>map.toList)

val Point = longitude.join(latitude)
val filteredPoint = Point.map(tp=>(tp._2._1,tp._2._2)).
					filter(tp=>tp._1 > -74.05 && tp._1 < -73.75 && tp._2 > 40.55 && tp._2 < 40.95)

val xy = filteredPoint.map(tp=>Vectors.dense(tp._1,tp._2)).cache()

val K = 20
val clusters = KMeans.train(xy, K, 10000)
val WSSSE = clusters.computeCost(xy)
clusters.save(sc, "rent/KMeansModel")
val clusteringModel = KMeansModel.load(sc, "rent/KMeansModel")

def get_neighborhood(p:(Double,Double), cluster:KMeansModel): Int={
	val x = p._1
	val y = p._2
	if (x > -74.05 && x < -73.75 && y > 40.55 && y < 40.95){
		return cluster.predict(Vectors.dense(x,y))
	}
	else return K
}
val neighborhood = Point.mapValues(p=>get_neighborhood(p,clusteringModel))

def building_scoring(label: List[Double]): Double = {
	val l_0 = label.count(_==0)
	val l_1 = label.count(_==1)
	val l_2 = label.count(_==2)
	val sum = l_0+l_1+l_2
	return (l_1 + 2*l_2).toDouble/sum
} 

//get building_id
val building_id = sc.wholeTextFiles("rent/train/train_building_id").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)

val building_score = building_id.join(train_label).map(tp=>(tp._2._1,tp._2._2)).groupByKey().
					   mapValues(x=>building_scoring(x.toList))

val bscoresum = building_score.map(tp=>tp._2).reduce(_+_)
val bmeanscore = bscoresum/building_score.count()
val bscoreMap = building_score.collect.toMap
sc.parallelize(bscoreMap.toSeq).saveAsTextFile("rent/building_score")
val building_quality = building_id.mapValues(x=>bscoreMap.getOrElse(x,bmeanscore))

//--------------listing info--------------
//get listing_id					
val listing_id = sc.wholeTextFiles("rent/train/train_listing_id").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,Double]]).
					flatMap(map=>map.toList)

//get train_manager_id
val manager_id = sc.wholeTextFiles("rent/train/train_manager_id").
					map(pair=>JSON.parseFull(pair._2).get.asInstanceOf[Map[String,String]]).
					flatMap(map=>map.toList)

def maga_id_encoding(label: List[Double]): Int = {
	val l_0 = label.count(_==0)
	val l_1 = label.count(_==1)
	val l_2 = label.count(_==2)
	if (l_1==0 && l_2==0) return 0 //0:0 
	else if (l_0==0 && l_2==0) return 6//1:6
	else if (l_0==0 && l_1==0) return 11//2:11
	else if (l_2==0){//0,1
		if (l_0>l_1) return 1 //0>1:1
		else   		return 4  //1>0:4
	}else if (l_1==0){//0,2
		if (l_0>l_2) return 3 //0>2:3
		else 		return 8  //2>0:8
	}else if (l_0==0){//1,2   
		if (l_1>l_2) return 7  //1>2:7
		else 		return  9   //2>1:9
	}else if(l_0>l_1 && l_0>l_2){//0>(1,2):2
		return  2
	}else if (l_1>l_0 && l_1>l_2){//1>(0,2):5
		return  5
	}else//2>(0,1):10
		return  10
}

def maga_skill_score(label: List[Double]): Double = {
	val l_0 = label.count(_==0)
	val l_1 = label.count(_==1)
	val l_2 = label.count(_==2)
	val sum = l_0+l_1+l_2
	return (l_1 + 2*l_2).toDouble/sum
}

val maga_label = manager_id.join(train_label).map(tp=>(tp._2._1,tp._2._2)).groupByKey()
maga_label.cache()
val maga_code = maga_label.mapValues(x=>maga_id_encoding(x.toList)).collect.toMap
sc.parallelize(maga_code.toSeq).saveAsTextFile("rent/manager_code")

val maga_score = maga_label.mapValues(x=>maga_skill_score(x.toList))

def getMagaCode(id: String, encode:Map[String,Int]): Int = {
	encode.getOrElse(id, -1)
}
val scoresum = maga_score.map(tp=>tp._2).reduce(_+_)
val meanscore = scoresum/maga_score.count()
val scoreMap = maga_score.collect.toMap
sc.parallelize(scoreMap.toSeq).saveAsTextFile("rent/manager_score")
val manager_feature = manager_id.mapValues(x=>getMagaCode(x,maga_code).toDouble)
val manager_skill = manager_id.mapValues(x=>scoreMap.getOrElse(x,meanscore))

//get created time
val created = sc.wholeTextFiles("rent/train/train_created").
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

val train_data = features.join(train_label).map(tp=>(tp._2._2,tp._2._1)).sortByKey().
					map(tp=>LabeledPoint(tp._1.toDouble, Vectors.dense(tp._2.toArray).toSparse))

MLUtils.saveAsLibSVMFile(train_data,"rent/train_data")
